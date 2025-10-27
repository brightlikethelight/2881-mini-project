#!/bin/bash
#
# Complete reproduction script using Together.ai API
# Reproduces Table 1 from "Follow My Instruction and Spill the Beans"
#
# Usage: ./reproduce_with_together.sh
#
# Prerequisites:
#   1. conda activate rag-privacy
#   2. pip install together
#   3. Create keys/mine.txt with Together API key
#
# Total time: ~60 minutes
# Total cost: ~$0.10

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo ""
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=====================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check prerequisites
print_step "[0/8] Checking Prerequisites"

if ! python -c "import together" 2>/dev/null; then
    print_error "Together AI package not installed"
    echo "Run: pip install together"
    exit 1
fi
print_success "Together AI package installed"

if [ ! -f "keys/mine.txt" ]; then
    print_error "API keys not found at keys/mine.txt"
    echo "Create it with: mkdir -p keys && echo 'YOUR_KEY' > keys/mine.txt"
    exit 1
fi
print_success "API keys found"

# Test API
print_step "[1/8] Testing Together API Connection"
python test_together_api.py || exit 1

# Generate prompts
print_step "[2/8] Generating Attack Prompts"
if [ -f "prompts/attack_prompts.json" ]; then
    print_warning "prompts/attack_prompts.json already exists, skipping generation"
else
    python scripts/generate_prompts.py \
        --num_samples 100 \
        --output prompts/attack_prompts.json
    print_success "Generated 100 attack prompts"
fi

# Define models
declare -A MODELS
MODELS=(
    ["Llama-2-7B"]="meta-llama/Llama-2-7b-chat-hf|meta-llama/Llama-2-7b-chat-hf"
    ["Llama-2-13B"]="meta-llama/Llama-2-13b-chat-hf|meta-llama/Llama-2-13b-chat-hf"
    ["Llama-2-70B"]="meta-llama/Llama-2-70b-chat-hf|meta-llama/Llama-2-70b-chat-hf"
    ["Mistral-7B"]="mistralai/Mistral-7B-Instruct-v0.1|mistralai/Mistral-7B-Instruct-v0.1"
    ["Qwen-7B"]="Qwen/Qwen-7B-Chat|Qwen/Qwen1.5-7B-Chat"
)

# Run inference for each model
STEP=3
for model_name in "Llama-2-7B" "Llama-2-13B" "Llama-2-70B" "Mistral-7B" "Qwen-7B"; do
    print_step "[$STEP/8] Running $model_name Inference"

    IFS='|' read -r hf_ckpt together_ckpt <<< "${MODELS[$model_name]}"

    # Check if outputs already exist
    model_output_dir="eval_data/outputs/$(basename $hf_ckpt)"
    if [ -d "$model_output_dir" ]; then
        num_outputs=$(ls "$model_output_dir"/*.json 2>/dev/null | wc -l)
        if [ "$num_outputs" -ge 100 ]; then
            print_warning "$model_name outputs already exist ($num_outputs files), skipping"
            ((STEP++))
            continue
        fi
    fi

    start_time=$(date +%s)

    python main.py --task io \
        --api together \
        --hf_ckpt "$hf_ckpt" \
        --together_ckpt "$together_ckpt" \
        --is_chat_model true \
        --raw_data_dir raw_data/private/wiki_newest \
        --io_input_path prompts/attack_prompts.json \
        --io_output_root eval_data/outputs \
        --datastore_root datastore \
        --output_dir out

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    print_success "$model_name inference completed in ${elapsed}s"
    ((STEP++))
done

# Evaluation
print_step "[$STEP/8] Evaluating All Models"
python main.py --task eval \
    --eval_input_dir eval_data/outputs \
    --eval_output_dir eval_data/results
print_success "Evaluation completed"
((STEP++))

# Generate table
print_step "[$STEP/8] Generating Results Table"
python scripts/generate_results_table.py \
    --results_dir eval_data/results
print_success "Results table generated"

# Display results
echo ""
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}REPRODUCTION COMPLETE!${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""
echo "Results are in:"
echo "  - eval_data/results/table.md (Markdown)"
echo "  - eval_data/results/table.tex (LaTeX)"
echo ""
echo "Preview:"
echo ""
cat eval_data/results/table.md
echo ""

print_success "All done! Check eval_data/results/ for full results."
