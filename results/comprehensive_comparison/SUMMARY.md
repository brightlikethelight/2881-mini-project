# Comprehensive Summarization Methods Comparison

## Overview
This analysis compares **6 different summarization methods** across **3 compression ratios** (30%, 50%, 70%):

### Methods Tested
1. **TextRank** - Graph-based extractive (Jaccard similarity)
2. **LexRank** - Graph-based extractive (TF-IDF cosine similarity)
3. **SumBasic** - Frequency-based extractive
4. **LSA/SVD** - Latent semantic analysis extractive
5. **MMR** - Maximal Marginal Relevance extractive
6. **LLM** - Abstractive (using Llama-3.2-3B)

### Compression Ratios
- **30%** - Aggressive compression (keep 30% of content)
- **50%** - Moderate compression (keep 50% of content)
- **70%** - Light compression (keep 70% of content)

---

## 🏆 Top Performers

### Best Utility (ROUGE-L)
**LexRank 70%** - Score: 0.772
- Best at preserving key information
- Uses TF-IDF weighted cosine similarity for sentence ranking
- Ideal when content fidelity is critical

### Best Privacy
**LLM 30%** - Score: 0.689
- Highest privacy protection through abstractive generation
- Only 83.5% token overlap (vs 100% for extractive methods)
- Generates new phrasing, reducing information leakage

### Best Compression
**MMR 30%** - Ratio: 0.251
- Achieves most aggressive compression (25.1% of original)
- Balances relevance and redundancy reduction
- Excellent for minimizing data exposure

### Best Overall Balance
**MMR 30%** - Utility: 0.391, Privacy: 0.666, Compression: 0.251
- Optimal trade-off between all metrics
- Maximal Marginal Relevance reduces redundancy effectively
- Good choice for privacy-sensitive applications

---

## 📊 Key Findings

### 1. Extractive vs Abstractive Methods

| Aspect | Extractive (TextRank, LexRank, etc.) | Abstractive (LLM) |
|--------|--------------------------------------|-------------------|
| **Token Overlap** | 100% | ~84-87% |
| **Privacy Score** | 0.33-0.69 | 0.54-0.69 |
| **Utility (ROUGE-L)** | 0.39-0.77 | 0.33-0.60 |
| **Compression** | 0.25-0.64 | 0.29-0.67 |

**Insight**: LLM methods provide better privacy through paraphrasing but may sacrifice some utility.

### 2. Performance by Compression Ratio

#### 30% Compression (Aggressive)
- **Best Utility**: LexRank (0.548)
- **Best Privacy**: LLM (0.689)
- **Recommendation**: Use LLM for privacy-critical, LexRank for utility-critical

#### 50% Compression (Moderate)
- **Best Utility**: LexRank (0.730)
- **Best Privacy**: LSA (0.470)
- **Recommendation**: LexRank offers excellent balance

#### 70% Compression (Light)
- **Best Utility**: LexRank (0.772)
- **Best Privacy**: MMR (0.666)
- **Recommendation**: LexRank for high utility, MMR for privacy

### 3. Method-Specific Insights

#### TextRank
- **Strengths**: Simple, interpretable, good baseline
- **Weaknesses**: Lower privacy scores
- **Best Use**: General-purpose summarization

#### LexRank
- **Strengths**: Highest utility across all ratios
- **Weaknesses**: Moderate privacy
- **Best Use**: When preserving information is critical

#### SumBasic
- **Strengths**: Fast, frequency-based, simple
- **Weaknesses**: Middle-of-the-road performance
- **Best Use**: Quick summaries, low computational cost

#### LSA/SVD
- **Strengths**: Captures semantic relationships
- **Weaknesses**: Variable performance
- **Best Use**: Topic-based summarization

#### MMR
- **Strengths**: Best compression, reduces redundancy
- **Weaknesses**: Lower utility at high compression
- **Best Use**: Minimizing data exposure, avoiding repetition

#### LLM
- **Strengths**: Best privacy, generates new text
- **Weaknesses**: Lower utility, API costs, slower
- **Best Use**: Privacy-sensitive applications

---

## 📈 Visualizations Generated

1. **heatmap_all_methods.png**
   - Color-coded comparison of all methods and metrics
   - Easy to spot best/worst performers

2. **privacy_utility_scatter.png**
   - Shows privacy-utility trade-off for all methods
   - Point size indicates compression ratio
   - Identifies optimal trade-off points

3. **compression_comparison.png**
   - Bar chart of actual compression achieved
   - Compares target vs actual compression

4. **comparison_by_ratio.png**
   - Side-by-side comparison at each compression level
   - Shows utility vs privacy for each ratio

5. **radar_charts.png**
   - Multi-metric radar plots for each compression ratio
   - Visualizes method strengths/weaknesses

6. **pareto_frontier.png**
   - Identifies Pareto-optimal methods
   - Shows methods that cannot be improved without trade-offs

---

## 💡 Recommendations by Use Case

### Maximum Privacy (Sensitive Data)
**Recommended**: LLM 30%
- Privacy Score: 0.689
- Utility: 0.333
- Use when: Protecting PII, medical records, financial data

### Maximum Utility (Information Preservation)
**Recommended**: LexRank 70%
- Utility: 0.772
- Privacy Score: 0.334
- Use when: Academic summaries, news articles, documentation

### Balanced Approach
**Recommended**: LLM 50% or MMR 50%
- LLM 50%: Utility 0.490, Privacy 0.538
- MMR 50%: Utility 0.391, Privacy 0.666
- Use when: General-purpose summarization with privacy awareness

### Minimal Storage/Transmission
**Recommended**: MMR 30%
- Compression: 0.251 (smallest)
- Privacy: 0.666
- Use when: Bandwidth constraints, storage limits

### Fast Processing
**Recommended**: SumBasic or TextRank
- No complex calculations
- Deterministic results
- Use when: Real-time summarization, high throughput

---

## 🔬 Technical Details

### Metrics Explained

1. **ROUGE-L (Utility)**
   - Measures longest common subsequence with reference
   - Higher = better information preservation
   - Range: 0-1

2. **Privacy Score**
   - Calculated as 1 - Jaccard Similarity
   - Higher = less overlap with original text
   - Range: 0-1

3. **Compression Ratio**
   - Actual words in summary / words in original
   - Lower = more compression
   - Range: 0-1

4. **Token Overlap**
   - Percentage of summary tokens from original
   - Lower = better privacy (less direct copying)
   - Range: 0-1

### Dataset
- Source: Wikipedia articles
- Samples: 5 paragraphs
- Average length: ~200 words per paragraph

---

## 🎯 Conclusion

The choice of summarization method depends on your specific requirements:

- **For Privacy**: Choose LLM-based methods
- **For Utility**: Choose LexRank
- **For Compression**: Choose MMR
- **For Speed**: Choose SumBasic or TextRank
- **For Balance**: Choose MMR at 30-50% or LLM at 50%

The Pareto frontier analysis shows that no single method dominates all others, confirming that the optimal choice depends on your specific privacy-utility-compression trade-off preferences.
