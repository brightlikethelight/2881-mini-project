# Extension 1: Prompt-Level Defenses - Results (FIXED)

## Leakage Metrics (Lower = Better Privacy)

| Config | ROUGE-L ↓ | Verbatim 5g ↓ | Verbatim 4g ↓ | Verbatim 3g ↓ | Avg Length | N |
|--------|-----------|---------------|---------------|---------------|------------|---|
| combined_light       | 0.510 | 0.514 | 0.550 | 0.599 | 262.2 | 50 |
| ngram_2              | 0.516 | 0.504 | 0.540 | 0.588 | 270.4 | 50 |
| ngram_4              | 0.524 | 0.511 | 0.549 | 0.596 | 277.4 | 50 |
| enc_ngram_4          | 0.524 | 0.527 | 0.563 | 0.610 | 271.5 | 50 |
| baseline             | 0.525 | 0.512 | 0.548 | 0.593 | 283.1 | 50 |
| combined_max         | 0.527 | 0.520 | 0.557 | 0.605 | 271.3 | 50 |
| enc_ngram_5          | 0.531 | 0.523 | 0.558 | 0.605 | 272.4 | 50 |
| combined_medium      | 0.531 | 0.523 | 0.558 | 0.605 | 274.1 | 50 |
| combined_strong      | 0.531 | 0.537 | 0.572 | 0.617 | 266.6 | 50 |
| bad_words_5          | 0.534 | 0.523 | 0.560 | 0.609 | 271.4 | 50 |
| ngram_3              | 0.537 | 0.528 | 0.563 | 0.610 | 271.4 | 50 |
| bad_words            | 0.538 | 0.529 | 0.566 | 0.614 | 272.9 | 50 |
| ngram_5              | 0.544 | 0.548 | 0.582 | 0.626 | 270.4 | 50 |
| enc_ngram_3          | 0.548 | 0.557 | 0.592 | 0.636 | 278.8 | 50 |
| enc_ngram_2          | 0.550 | 0.557 | 0.591 | 0.637 | 278.2 | 50 |

## Metrics Explanation
- **ROUGE-L**: Longest common subsequence overlap with retrieved docs (0-1, lower = less leakage)
- **Verbatim N-gram**: % of output that directly copies N-grams from docs (0-1, lower = less copying)
- **Avg Length**: Average output length in tokens
- ↓ = lower is better (less privacy leakage)

## Key Insights
- **Best defense** (lowest leakage): combined_light
- **Worst defense** (highest leakage): enc_ngram_2