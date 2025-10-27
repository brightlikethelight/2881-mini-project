#!/bin/bash
# Run Spanish RAG copy-out attack experiment
# For Cross-Lingual RAG Copy-Out Attack Experiment
# Based on ICLR 2025: "Follow My Instruction and Spill the Beans"

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default parameters (can be overridden by environment variables)
API="${API:-hf}"
RAW_DATA_DIR="${RAW_DATA_DIR:-./raw_data/private/wiki_spanish}"
IO_OUTPUT_ROOT="${IO_OUTPUT_ROOT:-./eval_data/spanish/io_output}"
DATASTORE_ROOT="${DATASTORE_ROOT:-./datastore}"
OUTPUT_DIR="${OUTPUT_DIR:-./out}"

# Models to test (mix of multilingual and English-centric)
MODELS=(
    "meta-llama/Llama-2-7b-chat-hf"      # Has some multilingual capability
    "Qwen/Qwen-7B-Chat"                  # Known multilingual model
    "mistralai/Mistral-7B-Instruct-v0.1" # English-centric for comparison
)

# Prompt types to test
PROMPT_TYPES=(
    "spanish"     # Native Spanish prompts
    "english"     # English prompts on Spanish datastore
    "codeswitch"  # Mixed English-Spanish prompts
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

    # Check if Spanish data exists
    if [ ! -d "$RAW_DATA_DIR" ]; then
        print_msg "$RED" "✗ Error: Spanish data directory not found: $RAW_DATA_DIR"
        print_msg "$YELLOW" "  Fetch with: python scripts/fetch_wikipedia_spanish.py"
        exit 1
    fi
    print_msg "$GREEN" "✓ Spanish data directory found: $RAW_DATA_DIR"

    # Check if prompt files exist
    for prompt_type in "${PROMPT_TYPES[@]}"; do
        prompt_file="prompts/attack_prompts_${prompt_type}.json"
        if [ ! -f "$prompt_file" ]; then
            print_msg "$RED" "✗ Error: Prompt file not found: $prompt_file"
            print_msg "$YELLOW" "  Generate with: python scripts/generate_prompts_spanish.py --prompt_type $prompt_type"
            exit 1
        fi
        print_msg "$GREEN" "✓ Prompt file found: $prompt_file"
    done

    # Check CUDA availability (optional)
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
        print_msg "$GREEN" "✓ CUDA available: $GPU_COUNT GPU(s)"
    else
        print_msg "$YELLOW" "⚠ CUDA not available, will use CPU (slower)"
    fi

    echo ""
}

# Estimate time for model
estimate_time() {
    local model_name=$1
    local prompt_type=$2

    # Rough estimates (adjust based on your hardware)
    if [[ $model_name == *"70b"* ]]; then
        echo "~4-6 hours (70B model, very slow)"
    elif [[ $model_name == *"13b"* ]]; then
        echo "~1-2 hours (13B model)"
    elif [[ $model_name == *"7b"* ]] || [[ $model_name == *"7B"* ]]; then
        if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            echo "~15-30 minutes (7B model, GPU)"
        else
            echo "~1-2 hours (7B model, CPU)"
        fi
    else
        echo "~30-60 minutes"
    fi
}

# Check if experiment is already complete
is_experiment_complete() {
    local model_name=$1
    local prompt_type=$2
    local model_short=$(basename "$model_name")
    local experiment_dir="$IO_OUTPUT_ROOT/${model_short}_${prompt_type}"

    if [ ! -d "$experiment_dir" ]; then
        return 1  # Not complete
    fi

    # Count JSON files
    local json_count=$(find "$experiment_dir" -name "*.json" 2>/dev/null | wc -l)
    local expected_count=$(python -c "import json; print(len(json.load(open('prompts/attack_prompts_${prompt_type}.json'))))" 2>/dev/null || echo "0")

    if [ "$json_count" -ge "$expected_count" ]; then
        return 0  # Complete
    else
        return 1  # Not complete
    fi
}

# Run single experiment
run_experiment() {
    local model=$1
    local prompt_type=$2
    local model_short=$(basename "$model")
    local prompt_file="prompts/attack_prompts_${prompt_type}.json"
    local experiment_dir="$IO_OUTPUT_ROOT/${model_short}_${prompt_type}"

    echo ""
    print_msg "$BLUE" "========================================="
    print_msg "$BLUE" "Model: $model"
    print_msg "$BLUE" "Prompt Type: $prompt_type"
    print_msg "$BLUE" "========================================="

    # Check if already complete
    if is_experiment_complete "$model" "$prompt_type"; then
        print_msg "$YELLOW" "⚠ Experiment already completed, skipping..."
        print_msg "$YELLOW" "  (Delete $experiment_dir to rerun)"
        return 0
    fi

    # Estimate time
    local est_time=$(estimate_time "$model" "$prompt_type")
    print_msg "$BLUE" "Estimated time: $est_time"
    echo ""

    # Create log file
    local log_file="$OUTPUT_DIR/logs/${model_short}_${prompt_type}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "$log_file")"

    # Run main.py
    print_msg "$GREEN" "Starting experiment..."
    print_msg "$BLUE" "Log file: $log_file"

    if python "$PROJECT_ROOT/main.py" \
        --task io \
        --api "$API" \
        --hf_ckpt "$model" \
        --is_chat_model true \
        --raw_data_dir "$RAW_DATA_DIR" \
        --io_input_path "$prompt_file" \
        --io_output_root "$experiment_dir" \
        --datastore_root "$DATASTORE_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "$log_file"; then

        print_msg "$GREEN" "✓ Completed: $model ($prompt_type)"
        return 0
    else
        print_msg "$RED" "✗ Failed: $model ($prompt_type)"
        print_msg "$RED" "  Check log: $log_file"
        return 1
    fi
}

# Main execution
main() {
    print_msg "$GREEN" "Spanish RAG Copy-Out Attack Experiment"
    print_msg "$GREEN" "Cross-Lingual Generalizability Test"
    echo ""

    # Check prerequisites
    check_prerequisites

    # Print configuration
    print_msg "$BLUE" "Configuration:"
    echo "  API: $API"
    echo "  Spanish data: $RAW_DATA_DIR"
    echo "  Output: $IO_OUTPUT_ROOT"
    echo "  Datastore: $DATASTORE_ROOT"
    echo "  Models: ${#MODELS[@]}"
    echo "  Prompt types: ${#PROMPT_TYPES[@]}"
    echo ""

    # Count total experiments
    total_experiments=$((${#MODELS[@]} * ${#PROMPT_TYPES[@]}))
    experiments_todo=0

    for model in "${MODELS[@]}"; do
        for prompt_type in "${PROMPT_TYPES[@]}"; do
            if ! is_experiment_complete "$model" "$prompt_type"; then
                ((experiments_todo++))
            fi
        done
    done

    print_msg "$YELLOW" "Experiments to run: $experiments_todo / $total_experiments"
    echo ""

    # Confirm before starting
    if [ "$experiments_todo" -eq 0 ]; then
        print_msg "$GREEN" "All experiments already completed!"
        print_msg "$YELLOW" "To rerun, delete output directories in: $IO_OUTPUT_ROOT"
        exit 0
    fi

    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_msg "$YELLOW" "Cancelled by user"
        exit 0
    fi

    # Process each experiment
    local success_count=0
    local fail_count=0
    local skip_count=0
    local start_time=$(date +%s)

    for model in "${MODELS[@]}"; do
        for prompt_type in "${PROMPT_TYPES[@]}"; do
            if is_experiment_complete "$model" "$prompt_type"; then
                ((skip_count++))
                print_msg "$YELLOW" "Skipping $(basename $model) ($prompt_type) - already complete"
                continue
            fi

            if run_experiment "$model" "$prompt_type"; then
                ((success_count++))
            else
                ((fail_count++))
                print_msg "$RED" "⚠ Warning: Experiment failed, continuing with next..."
            fi
        done
    done

    # Print summary
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))

    echo ""
    print_msg "$BLUE" "========================================="
    print_msg "$BLUE" "SUMMARY"
    print_msg "$BLUE" "========================================="
    print_msg "$GREEN" "✓ Successful: $success_count"
    print_msg "$RED" "✗ Failed: $fail_count"
    print_msg "$YELLOW" "⊘ Skipped: $skip_count"
    print_msg "$BLUE" "Total time: ${hours}h ${minutes}m"
    echo ""

    if [ $fail_count -eq 0 ]; then
        print_msg "$GREEN" "All experiments completed successfully!"
        echo ""
        print_msg "$BLUE" "Next steps:"
        print_msg "$BLUE" "  1. Run evaluation: python main.py --task eval --eval_input_dir $IO_OUTPUT_ROOT --eval_output_dir ./eval_data/spanish/eval_results"
        print_msg "$BLUE" "  2. Generate Spanish results: python scripts/generate_spanish_results.py"
    else
        print_msg "$YELLOW" "Some experiments failed. Check logs in: $OUTPUT_DIR/logs/"
    fi

    echo ""
}

# Handle Ctrl+C gracefully
trap 'echo ""; print_msg "$YELLOW" "Interrupted by user"; exit 130' INT

# Run main
main "$@"


