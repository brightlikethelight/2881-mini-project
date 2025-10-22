#!/bin/bash
# Run RAG copy-out attack on multiple models
# For ICLR 2025: "Follow My Instruction and Spill the Beans"

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default parameters (can be overridden by environment variables)
API="${API:-hf}"
RAW_DATA_DIR="${RAW_DATA_DIR:-./raw_data/wikipedia_nov2023}"
IO_INPUT_PATH="${IO_INPUT_PATH:-./prompts/attack_prompts.json}"
IO_OUTPUT_ROOT="${IO_OUTPUT_ROOT:-./eval_data/wikipedia/io_output}"
DATASTORE_ROOT="${DATASTORE_ROOT:-./datastore}"
OUTPUT_DIR="${OUTPUT_DIR:-./out}"

# Models to test (from paper)
MODELS=(
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-2-13b-chat-hf"
    "mistralai/Mistral-7B-Instruct-v0.1"
    "Qwen/Qwen-7B-Chat"
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
    local num_prompts=$(python -c "import json; print(len(json.load(open('$IO_INPUT_PATH'))))" 2>/dev/null || echo "100")

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

# Check if model is already processed
is_model_complete() {
    local model_name=$1
    local model_output_dir="$IO_OUTPUT_ROOT/$(basename $model_name)"

    if [ ! -d "$model_output_dir" ]; then
        return 1  # Not complete
    fi

    # Count JSON files
    local json_count=$(find "$model_output_dir" -name "*.json" 2>/dev/null | wc -l)
    local expected_count=$(python -c "import json; print(len(json.load(open('$IO_INPUT_PATH'))))" 2>/dev/null || echo "0")

    if [ "$json_count" -ge "$expected_count" ]; then
        return 0  # Complete
    else
        return 1  # Not complete
    fi
}

# Run inference for single model
run_model() {
    local model=$1
    local model_short=$(basename "$model")

    echo ""
    print_msg "$BLUE" "========================================="
    print_msg "$BLUE" "Model: $model"
    print_msg "$BLUE" "========================================="

    # Check if already complete
    if is_model_complete "$model"; then
        print_msg "$YELLOW" "⚠ Model already processed, skipping..."
        print_msg "$YELLOW" "  (Delete $IO_OUTPUT_ROOT/$model_short to rerun)"
        return 0
    fi

    # Estimate time
    local est_time=$(estimate_time "$model")
    print_msg "$BLUE" "Estimated time: $est_time"
    echo ""

    # Create log file
    local log_file="$OUTPUT_DIR/logs/${model_short}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "$log_file")"

    # Run main.py
    print_msg "$GREEN" "Starting inference..."
    print_msg "$BLUE" "Log file: $log_file"

    if python "$PROJECT_ROOT/main.py" \
        --task io \
        --api "$API" \
        --hf_ckpt "$model" \
        --is_chat_model true \
        --raw_data_dir "$RAW_DATA_DIR" \
        --io_input_path "$IO_INPUT_PATH" \
        --io_output_root "$IO_OUTPUT_ROOT" \
        --datastore_root "$DATASTORE_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "$log_file"; then

        print_msg "$GREEN" "✓ Completed: $model"
        return 0
    else
        print_msg "$RED" "✗ Failed: $model"
        print_msg "$RED" "  Check log: $log_file"
        return 1
    fi
}

# Main execution
main() {
    print_msg "$GREEN" "RAG Copy-Out Attack - Batch Model Runner"
    print_msg "$GREEN" "For ICLR 2025: Follow My Instruction and Spill the Beans"
    echo ""

    # Check prerequisites
    check_prerequisites

    # Print configuration
    print_msg "$BLUE" "Configuration:"
    echo "  API: $API"
    echo "  Raw data: $RAW_DATA_DIR"
    echo "  Prompts: $IO_INPUT_PATH"
    echo "  Output: $IO_OUTPUT_ROOT"
    echo "  Datastore: $DATASTORE_ROOT"
    echo "  Models: ${#MODELS[@]}"
    echo ""

    # Count how many models need processing
    models_todo=0
    for model in "${MODELS[@]}"; do
        if ! is_model_complete "$model"; then
            ((models_todo++))
        fi
    done

    print_msg "$YELLOW" "Models to process: $models_todo / ${#MODELS[@]}"
    echo ""

    # Confirm before starting
    if [ "$models_todo" -eq 0 ]; then
        print_msg "$GREEN" "All models already processed!"
        print_msg "$YELLOW" "To rerun, delete output directories in: $IO_OUTPUT_ROOT"
        exit 0
    fi

    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_msg "$YELLOW" "Cancelled by user"
        exit 0
    fi

    # Process each model
    local success_count=0
    local fail_count=0
    local skip_count=0
    local start_time=$(date +%s)

    for model in "${MODELS[@]}"; do
        if is_model_complete "$model"; then
            ((skip_count++))
            print_msg "$YELLOW" "Skipping $(basename $model) (already complete)"
            continue
        fi

        if run_model "$model"; then
            ((success_count++))
        else
            ((fail_count++))
            print_msg "$RED" "⚠ Warning: Model failed, continuing with next model..."
        fi
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
        print_msg "$GREEN" "All models completed successfully!"
        echo ""
        print_msg "$BLUE" "Next steps:"
        print_msg "$BLUE" "  1. Run evaluation: python main.py --task eval --eval_input_dir $IO_OUTPUT_ROOT --eval_output_dir ./eval_data/wikipedia/eval_results"
        print_msg "$BLUE" "  2. Generate tables: python scripts/generate_results_table.py"
    else
        print_msg "$YELLOW" "Some models failed. Check logs in: $OUTPUT_DIR/logs/"
    fi

    echo ""
}

# Handle Ctrl+C gracefully
trap 'echo ""; print_msg "$YELLOW" "Interrupted by user"; exit 130' INT

# Run main
main "$@"
