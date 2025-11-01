#!/bin/bash
# Run RAG copy-out attack on models from Table 1 (7b and 13b rows)
# Automatically uses Together.ai API for supported models, local HF for others

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default parameters (can be overridden by environment variables)
RAW_DATA_DIR="${RAW_DATA_DIR:-./raw_data/private/wiki_newest}"
IO_INPUT_PATH="${IO_INPUT_PATH:-./prompts/attack_prompts.json}"
IO_OUTPUT_ROOT="${IO_OUTPUT_ROOT:-./eval_data/wikipedia/io_output}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-./eval_data/wikipedia/eval_results}"
DATASTORE_ROOT="${DATASTORE_ROOT:-./datastore}"
OUTPUT_DIR="${OUTPUT_DIR:-./out}"

# Model configurations: "HF_MODEL_ID|TOGETHER_MODEL_ID|API|IS_CHAT"
# Format: Use "hf" API for local models, "together" for Together.ai API
MODELS=(
    # 7b models
    "meta-llama/Llama-2-7b-chat-hf|meta-llama/Llama-2-7b-chat-hf|hf|true"
    # "mistralai/Mistral-7B-Instruct-v0.1|mistralai/Mistral-7B-Instruct-v0.1|together|true"
    "upstage/SOLAR-10.7B-Instruct-v1.0|upstage/SOLAR-10.7B-Instruct-v1.0|together|true"

    # 13b models
    "meta-llama/Llama-2-13b-chat-hf|meta-llama/Llama-2-13b-chat-hf|together|true"
    # "lmsys/vicuna-13b-v1.5|lmsys/vicuna-13b-v1.5|hf|true"
    # "mistralai/Mixtral-8x7B-Instruct-v0.1|mistralai/Mixtral-8x7B-Instruct-v0.1|together|true"
    # "WizardLM/WizardLM-13B-V1.2|WizardLM/WizardLM-13B-V1.2|hf|true"
)

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_msg "$BLUE" "Checking prerequisites..."

    # Check if Python is available
    if ! command -v python &> /dev/null; then
        print_msg "$RED" "✗ Error: python not found"
        exit 1
    fi
    print_msg "$GREEN" "✓ Python found: $(python --version)"

    # Check if main.py exists
    if [ ! -f "$PROJECT_ROOT/main.py" ]; then
        print_msg "$RED" "✗ Error: main.py not found in $PROJECT_ROOT"
        exit 1
    fi
    print_msg "$GREEN" "✓ main.py found"

    # Check if input prompts exist
    if [ ! -f "$IO_INPUT_PATH" ]; then
        print_msg "$RED" "✗ Error: Prompts file not found: $IO_INPUT_PATH"
        print_msg "$YELLOW" "  Generate with: python scripts/generate_prompts.py"
        exit 1
    fi
    print_msg "$GREEN" "✓ Prompts file found: $IO_INPUT_PATH"

    # Check if raw data exists
    if [ ! -d "$RAW_DATA_DIR" ]; then
        print_msg "$RED" "✗ Error: Raw data directory not found: $RAW_DATA_DIR"
        print_msg "$YELLOW" "  Fetch with: python scripts/fetch_wikipedia.py"
        exit 1
    fi
    print_msg "$GREEN" "✓ Raw data directory found: $RAW_DATA_DIR"

    # Check CUDA availability (optional, needed for local models)
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
        print_msg "$GREEN" "✓ CUDA available: $GPU_COUNT GPU(s)"
    else
        print_msg "$YELLOW" "⚠ CUDA not available - only Together.ai API models will work"
    fi

    # Check Together.ai API keys (optional, needed for API models)
    if [ -f "$PROJECT_ROOT/keys/mine.txt" ]; then
        print_msg "$GREEN" "✓ Together.ai API keys found"
    else
        print_msg "$YELLOW" "⚠ Together.ai API keys not found at keys/mine.txt"
        print_msg "$YELLOW" "  Only local HF models will work"
    fi

    echo ""
}

# Run a single model
run_model() {
    local config=$1
    IFS='|' read -r hf_model together_model api is_chat <<< "$config"

    local model_short=$(echo "$hf_model" | awk -F'/' '{print $NF}')

    print_msg "$GREEN" "=================================================="
    print_msg "$GREEN" "Processing: $model_short"
    print_msg "$BLUE" "  HF Model: $hf_model"
    print_msg "$BLUE" "  Together Model: $together_model"
    print_msg "$BLUE" "  API: $api"
    print_msg "$BLUE" "  Is Chat: $is_chat"
    print_msg "$GREEN" "=================================================="

    # Check if already processed
    if [ -d "$IO_OUTPUT_ROOT/$model_short" ] && [ "$(ls -A "$IO_OUTPUT_ROOT/$model_short" 2>/dev/null)" ]; then
        print_msg "$YELLOW" "⚠ Model already processed, skipping..."
        print_msg "$YELLOW" "  (Delete $IO_OUTPUT_ROOT/$model_short to rerun)"
        return 0
    fi

    # Create log file
    local log_file="$OUTPUT_DIR/logs/${model_short}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "$log_file")"

    # Run main.py for IO task
    print_msg "$GREEN" "Starting inference (IO task)..."
    print_msg "$BLUE" "Log file: $log_file"

    if python "$PROJECT_ROOT/main.py" \
        --task io \
        --api "$api" \
        --hf_ckpt "$hf_model" \
        --together_ckpt "$together_model" \
        --is_chat_model "$is_chat" \
        --raw_data_dir "$RAW_DATA_DIR" \
        --io_input_path "$IO_INPUT_PATH" \
        --io_output_root "$IO_OUTPUT_ROOT" \
        --datastore_root "$DATASTORE_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "$log_file"; then

        print_msg "$GREEN" "✓ IO task completed: $model_short"
    else
        print_msg "$RED" "✗ IO task failed: $model_short"
        print_msg "$RED" "  Check log: $log_file"
        return 1
    fi
}

# Run evaluation on all models
run_eval() {
    print_msg "$GREEN" "=================================================="
    print_msg "$GREEN" "Running evaluation on all model outputs"
    print_msg "$GREEN" "=================================================="

    local eval_log="$OUTPUT_DIR/logs/eval_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "$eval_log")"

    if python "$PROJECT_ROOT/main.py" \
        --task eval \
        --eval_input_dir "$IO_OUTPUT_ROOT" \
        --eval_output_dir "$EVAL_OUTPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "$eval_log"; then

        print_msg "$GREEN" "✓ Evaluation completed"
        print_msg "$BLUE" "Results saved to: $EVAL_OUTPUT_DIR"
    else
        print_msg "$RED" "✗ Evaluation failed"
        print_msg "$RED" "  Check log: $eval_log"
        return 1
    fi
}

# Main execution
main() {
    print_msg "$GREEN" "RAG Copy-Out Attack - Table 1 Models (7b & 13b)"
    print_msg "$GREEN" "For ICLR 2025: Follow My Instruction and Spill the Beans"
    echo ""

    # Check prerequisites
    check_prerequisites

    # Print configuration
    print_msg "$BLUE" "Configuration:"
    print_msg "$BLUE" "  Raw Data: $RAW_DATA_DIR"
    print_msg "$BLUE" "  Prompts: $IO_INPUT_PATH"
    print_msg "$BLUE" "  Output: $IO_OUTPUT_ROOT"
    print_msg "$BLUE" "  Eval Output: $EVAL_OUTPUT_DIR"
    print_msg "$BLUE" "  Datastore: $DATASTORE_ROOT"
    echo ""

    # Print model list
    print_msg "$BLUE" "Models to process:"
    for i in "${!MODELS[@]}"; do
        local config="${MODELS[$i]}"
        IFS='|' read -r hf_model together_model api is_chat <<< "$config"
        local model_short=$(echo "$hf_model" | awk -F'/' '{print $NF}')
        local api_label="[$(echo $api | tr '[:lower:]' '[:upper:]')]"
        print_msg "$BLUE" "  $((i+1)). $model_short $api_label"
    done
    echo ""

    # Ask for confirmation
    read -p "Start processing? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_msg "$YELLOW" "Aborted by user"
        exit 0
    fi
    echo ""

    # Process each model
    local success_count=0
    local fail_count=0
    local start_time=$(date +%s)

    for config in "${MODELS[@]}"; do
        if run_model "$config"; then
            ((success_count++))
        else
            ((fail_count++))
            print_msg "$YELLOW" "⚠ Continuing with next model..."
        fi
        echo ""
    done

    # Run evaluation
    if [ $success_count -gt 0 ]; then
        run_eval
    fi

    # Print summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))

    print_msg "$GREEN" "=================================================="
    print_msg "$GREEN" "Summary"
    print_msg "$GREEN" "=================================================="
    print_msg "$GREEN" "✓ Successful: $success_count"
    print_msg "$RED" "✗ Failed: $fail_count"
    print_msg "$BLUE" "⏱ Total time: ${hours}h ${minutes}m ${seconds}s"
    print_msg "$GREEN" "=================================================="
}

# Run main
main
