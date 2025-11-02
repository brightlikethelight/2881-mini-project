# Extension 1: Prompt-Level Defenses - Results (FIXED)

## Leakage Metrics (Lower = Better Privacy)

| Config | ROUGE-L ↓ | Verbatim 5g ↓ | Verbatim 4g ↓ | Verbatim 3g ↓ | Avg Length | N |
|--------|-----------|---------------|---------------|---------------|------------|---|
| combined_max         | 0.514 | 0.473 | 0.501 | 0.536 | 309.2 | 50 |
| combined_light       | 0.515 | 0.481 | 0.510 | 0.548 | 299.1 | 50 |
| ngram_5              | 0.516 | 0.492 | 0.520 | 0.555 | 301.7 | 50 |
| enc_ngram_2          | 0.520 | 0.483 | 0.512 | 0.550 | 304.8 | 50 |
| enc_ngram_3          | 0.523 | 0.500 | 0.526 | 0.560 | 304.0 | 50 |
| bad_words            | 0.524 | 0.492 | 0.522 | 0.561 | 300.7 | 50 |
| enc_ngram_5          | 0.525 | 0.491 | 0.517 | 0.551 | 313.2 | 50 |
| ngram_2              | 0.527 | 0.503 | 0.531 | 0.567 | 300.6 | 50 |
| enc_ngram_4          | 0.528 | 0.504 | 0.531 | 0.566 | 302.8 | 50 |
| ngram_3              | 0.529 | 0.500 | 0.529 | 0.565 | 303.8 | 50 |
| ngram_4              | 0.532 | 0.510 | 0.538 | 0.573 | 307.6 | 50 |
| combined_medium      | 0.532 | 0.508 | 0.537 | 0.574 | 307.9 | 50 |
| baseline             | 0.532 | 0.503 | 0.532 | 0.569 | 305.9 | 50 |
| combined_strong      | 0.540 | 0.512 | 0.540 | 0.575 | 307.5 | 50 |
| bad_words_5          | 0.545 | 0.511 | 0.541 | 0.578 | 304.8 | 50 |

## Metrics Explanation
- **ROUGE-L**: Longest common subsequence overlap with retrieved docs (0-1, lower = less leakage)
- **Verbatim N-gram**: % of output that directly copies N-grams from docs (0-1, lower = less copying)
- **Avg Length**: Average output length in tokens
- ↓ = lower is better (less privacy leakage)

## Key Insights
- **Best defense** (lowest leakage): combined_max
- **Worst defense** (highest leakage): bad_words_5