# Extension 1: Prompt-Level Defenses

**Research Question:** Can we reduce verbatim copying from RAG systems using decoding constraints without significantly harming utility?

## Overview

This extension implements and evaluates three types of prompt-level defenses:

1. **No-Repeat N-gram**: Blocks repetition of n-grams within the generated output
2. **Encoder No-Repeat N-gram**: Blocks n-grams from the input context appearing in output
3. **Bad Words Defense**: Dynamically blacklists n-grams from retrieved documents

## Quick Start

### 1. Local Testing (gpt2, 10 prompts, 3 configs)

```bash
# Run defense sweep
python extensions/1_prompt_defenses/scripts/run_defense_sweep.py \
    --api hf \
    --hf_ckpt gpt2 \
    --num_prompts 10 \
    --configs baseline ngram_3 bad_words

# Analyze results
python extensions/1_prompt_defenses/scripts/analyze_defenses.py \
    --results_dir extensions/1_prompt_defenses/results
```

### 2. Full Experiments (Llama-2-7B, 50 prompts, all configs)

```bash
# Run defense sweep via Together API
python extensions/1_prompt_defenses/scripts/run_defense_sweep.py \
    --api together \
    --hf_ckpt meta-llama/Llama-2-7b-chat-hf \
    --together_ckpt togethercomputer/llama-2-7b-chat \
    --num_prompts 50 \
    --configs all

# Analyze results
python extensions/1_prompt_defenses/scripts/analyze_defenses.py
```

## Defense Configurations

| Config | no_repeat_ngram_size | encoder_no_repeat_ngram_size | use_bad_words_defense | Description |
|--------|---------------------|------------------------------|----------------------|-------------|
| baseline | 0 | 0 | False | No defense |
| ngram_2 | 2 | 0 | False | Block 2-gram repetition |
| ngram_3 | 3 | 0 | False | Block 3-gram repetition |
| ngram_4 | 4 | 0 | False | Block 4-gram repetition |
| ngram_5 | 5 | 0 | False | Block 5-gram repetition |
| enc_ngram_2 | 0 | 2 | False | Block 2-grams from input |
| enc_ngram_3 | 0 | 3 | False | Block 3-grams from input |
| enc_ngram_4 | 0 | 4 | False | Block 4-grams from input |
| enc_ngram_5 | 0 | 5 | False | Block 5-grams from input |
| bad_words | 0 | 0 | True (4-gram) | Blacklist 4-grams from docs |
| bad_words_5 | 0 | 0 | True (5-gram) | Blacklist 5-grams from docs |
| combined_light | 3 | 3 | False | Both ngram defenses (3-gram) |
| combined_medium | 4 | 4 | False | Both ngram defenses (4-gram) |
| combined_strong | 5 | 5 | False | Both ngram defenses (5-gram) |
| combined_max | 5 | 5 | True (5-gram) | All defenses at maximum |

## Metrics

- **Utility**: QA F1 (token-set F1 between prediction and gold answer)
- **Leakage**:
  - ROUGE-L (longest common subsequence overlap with retrieved docs)
  - Verbatim 5-gram rate (% of output that's verbatim from docs)
  - Verbatim 4-gram rate

## Expected Results

**Hypothesis:** There's a clean utility-leakage tradeoff curve:

- Baseline: High utility, high leakage
- Light defenses (ngram_2-3): Slight utility drop, moderate leakage reduction
- Strong defenses (ngram_4-5, bad_words): Larger utility drop, significant leakage reduction
- Combined defenses: Strongest leakage reduction, but utility may degrade substantially

**Key Findings to Look For:**

1. Which defense provides the best utility/leakage tradeoff?
2. Is `encoder_no_repeat_ngram_size` more effective than `no_repeat_ngram_size`?
3. Does `bad_words_defense` work better than ngram constraints?
4. Can we achieve <10% verbatim rate while maintaining >0.5 QA F1?

## Output Structure

```
extensions/1_prompt_defenses/results/
├── outputs/                      # Raw model outputs
│   ├── baseline/
│   │   └── results.json
│   ├── ngram_3/
│   │   └── results.json
│   └── ...
├── plots/                        # Analysis plots
│   ├── utility_vs_leakage.png
│   ├── defense_comparison.png
│   └── metrics.json
├── metrics_summary.md            # Summary table
└── summary.json                  # Experiment metadata
```

## Files

- `scripts/run_defense_sweep.py`: Run all defense configurations
- `scripts/analyze_defenses.py`: Compute metrics and generate plots
- `configs/`: (Optional) JSON configs for custom defense combinations

## Implementation Details

### Core Modifications

**utils/argparser.py (lines 34-39):**
```python
# Extension 1: Prompt-level defenses
no_repeat_ngram_size: int = field(default=0)
encoder_no_repeat_ngram_size: int = field(default=0)
use_bad_words_defense: bool = field(default=False)
bad_words_ngram_size: int = field(default=4)
defense_system_prompt: str = field(default=None)
```

**modules/LM.py (lines 34-35, 63-94, 107-110):**
- Added defense parameters to `GenerationConfig`
- Implemented `_extract_bad_words_ids()` helper method
- Modified `generate()` to accept `retrieved_docs_str` and extract bad words

**modules/RALM.py (line 65):**
```python
output_dict = self.lm.generate(lm_input, compute_generation_scores, compute_input_loss, retrieved_docs_str=docs_str)
```

## Cost Estimate

**Together API Costs (Llama-2-7B):**
- 15 configs × 50 prompts × ~500 tokens/prompt = ~375K tokens
- Input: 375K tokens × $0.20/1M = $0.075
- Output: 25K tokens × $0.20/1M = $0.005
- **Total: ~$0.08 per full run**

## Notes

- `encoder_no_repeat_ngram_size` requires HuggingFace Transformers ≥4.20.0
- `bad_words_ids` has a 10K limit to avoid memory issues
- Defenses only apply when `api=hf` (local models), not Together API
- For Together API testing, you'll need to implement defenses in post-processing

## Next Steps

1. Run full experiments with Llama-2-7B and Mistral-7B
2. Generate utility vs leakage plots
3. Write up findings in a research note
4. Compare with Extension 2 (retriever-based defenses)
