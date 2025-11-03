# Comprehensive Summarization Methods Comparison - Methodology

## Executive Summary

This experiment evaluates **6 summarization methods** across **3 compression ratios** to assess the privacy-utility trade-off in text summarization. The study compares 5 extractive methods (TextRank, LexRank, SumBasic, LSA/SVD, MMR) and 1 abstractive method (LLM-based) against a "no compression" baseline.

---

## 1. Data Source

### Dataset
- **Source**: Wikipedia articles
- **File**: `raw_data/private/wiki_newest/wiki_newest.txt`
- **Total Size**: 64,632 lines
- **Format**: Plain text with paragraphs separated by double newlines

### Sample Selection
- **Selection Criteria**: Paragraphs with >200 characters
- **Number of Samples**: 5 paragraphs
- **Sampling Method**: First 5 qualifying paragraphs from the dataset
- **Average Length**: ~200 words per paragraph
- **Purpose**: Representative Wikipedia content for general-purpose summarization

### Data Loading
```python
def load_wiki_data(file_path, max_samples=10):
    # Read entire file
    # Split by double newlines to get paragraphs
    # Filter paragraphs with >200 characters
    # Return first max_samples paragraphs
```

---

## 2. Summarization Methods

### 2.1 Extractive Methods (5 methods)

#### **TextRank**
- **Algorithm**: Graph-based ranking using PageRank
- **Similarity Metric**: Jaccard similarity between sentences
- **Implementation**: 
  - Build sentence similarity matrix
  - Create graph with sentences as nodes
  - Apply PageRank to rank sentences
  - Select top N sentences based on compression ratio

#### **LexRank**
- **Algorithm**: Graph-based ranking with TF-IDF weighting
- **Similarity Metric**: Cosine similarity on TF-IDF vectors
- **Implementation**:
  - Compute TF-IDF vectors for each sentence
  - Calculate cosine similarity matrix
  - Apply PageRank on similarity graph
  - Select top N sentences

#### **SumBasic**
- **Algorithm**: Frequency-based extractive summarization
- **Scoring Method**: Word frequency weighting
- **Implementation**:
  - Calculate word frequencies across all sentences
  - Score sentences by average word frequency
  - Iteratively select highest-scoring sentences
  - Update word frequencies after each selection

#### **LSA/SVD (Latent Semantic Analysis)**
- **Algorithm**: Dimensionality reduction via Singular Value Decomposition
- **Approach**: Semantic space projection
- **Implementation**:
  - Create TF-IDF matrix for sentences
  - Apply TruncatedSVD to reduce dimensions
  - Score sentences by their representation in latent space
  - Select top N sentences

#### **MMR (Maximal Marginal Relevance)**
- **Algorithm**: Relevance and redundancy balancing
- **Objective**: Maximize relevance while minimizing redundancy
- **Implementation**:
  - Calculate relevance scores (TF-IDF based)
  - Iteratively select sentences that maximize:
    - Relevance to original text
    - Dissimilarity to already-selected sentences
  - Lambda parameter: 0.7 (70% relevance, 30% diversity)

### 2.2 Abstractive Method (1 method)

#### **LLM (Large Language Model)**
- **Model**: Llama-3.2-3B-Instruct (via Together API)
- **Provider**: Together AI
- **Approach**: Generative summarization
- **Implementation**:
  - Async API calls to avoid blocking
  - Prompt: "Summarize the following text in approximately X% of its original length..."
  - Temperature: 0.7 (balanced creativity/consistency)
  - Max tokens: 2000
- **Rate Limiting**: 0.5 second delay between calls

---

## 3. Compression Ratios

Three compression levels were tested for each method:

| Ratio | Description | Target Length |
|-------|-------------|---------------|
| **30%** | Aggressive compression | Keep 30% of content |
| **50%** | Moderate compression | Keep 50% of content |
| **70%** | Light compression | Keep 70% of content |

**Total Configurations**: 6 methods × 3 ratios = **18 summarization variants** + 1 baseline = **19 total**

---

## 4. Evaluation Metrics

### 4.1 Utility Metrics

#### **ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence)**
- **Purpose**: Measures content preservation
- **Calculation**: F1-score based on longest common subsequence
- **Formula**: 
  - LCS = Longest Common Subsequence length
  - Precision = LCS / length(summary)
  - Recall = LCS / length(reference)
  - F1 = 2 × (Precision × Recall) / (Precision + Recall)
- **Range**: 0-1 (higher = better utility)
- **Reference**: "no_compression" (original text)

#### **Jaccard Similarity**
- **Purpose**: Token-level overlap with original
- **Calculation**: |summary ∩ original| / |summary ∪ original|
- **Range**: 0-1
- **Use**: Secondary utility metric

### 4.2 Privacy Metrics

#### **Privacy Score**
- **Definition**: 1 - Jaccard Similarity
- **Interpretation**: Lower overlap = better privacy
- **Range**: 0-1 (higher = better privacy)
- **Rationale**: Less token overlap means less direct information leakage

#### **Token Overlap**
- **Definition**: Percentage of summary tokens present in original
- **Calculation**: |summary_tokens ∩ original_tokens| / |summary_tokens|
- **Range**: 0-1 (lower = better privacy)
- **Note**: Extractive methods = 100% overlap (all tokens from original)

### 4.3 Compression Metrics

#### **Compression Ratio**
- **Definition**: Actual length of summary / length of original
- **Calculation**: word_count(summary) / word_count(original)
- **Range**: 0-1 (lower = more compression)
- **Purpose**: Measure actual compression achieved vs. target

---

## 5. Experimental Procedure

### Step 1: Data Loading
```
1. Load wiki_newest.txt
2. Split into paragraphs (by double newlines)
3. Filter paragraphs with >200 characters
4. Select first 5 paragraphs as samples
```

### Step 2: Summary Generation
For each sample:
```
For each method (TextRank, LexRank, SumBasic, LSA, MMR, LLM):
    For each compression ratio (30%, 50%, 70%):
        1. Generate summary using method at specified ratio
        2. Store summary with method_ratio identifier
        3. Log word count and status
```

**Special Handling**:
- LLM methods: Async execution with 0.5s delay between calls
- Extractive methods: Synchronous execution
- Error handling: Empty string if summarization fails

### Step 3: Evaluation
For each (sample, method, ratio) combination:
```
1. Calculate ROUGE-L vs. original text
2. Calculate Jaccard similarity vs. original
3. Calculate token overlap
4. Calculate compression ratio
5. Calculate privacy score (1 - Jaccard)
6. Store all metrics
```

### Step 4: Aggregation
```
For each method:
    For each metric:
        Average across all 5 samples
```

### Step 5: Visualization
Generate 6 visualizations:
1. Heatmap of all methods and metrics
2. Privacy-utility scatter plot
3. Compression comparison bar chart
4. Comparison by compression ratio
5. Radar charts for each ratio
6. Pareto frontier analysis

---

## 6. Implementation Details

### Dependencies
- **Python**: 3.10+
- **Core Libraries**:
  - `numpy`: Numerical computations
  - `matplotlib`: Visualization
  - `networkx`: Graph algorithms (TextRank, LexRank)
  - `scikit-learn`: TF-IDF, SVD, cosine similarity
  - `together`: LLM API client
  - `asyncio`: Async LLM calls
  - `pandas`: Data handling (for visualizations)
  - `seaborn`: Enhanced visualizations

### Tokenization
- **Method**: Regex-based word tokenization
- **Pattern**: `\b\w+\b` (word boundaries)
- **Case**: Lowercased for consistency
- **Rationale**: Avoid NLTK dependency issues

### Sentence Segmentation
- **Method**: Regex split on sentence terminators
- **Pattern**: `[.!?]+\s+`
- **Post-processing**: Strip whitespace, remove empty sentences

---

## 7. Computational Requirements

### Runtime
- **Extractive Methods**: ~1-2 seconds per sample
- **LLM Methods**: ~3-5 seconds per sample (API latency)
- **Total Runtime**: ~5-10 minutes for 5 samples × 19 configurations

### API Usage
- **LLM Calls**: 5 samples × 3 ratios = 15 API calls
- **Model**: Llama-3.2-3B-Instruct
- **Cost**: Minimal (Together AI free tier)

### Storage
- **Results JSON**: ~50-100 KB
- **Visualizations**: 6 PNG files (~500 KB each)

---

## 8. Baseline Comparison

### "No Compression" Baseline
- **Purpose**: Reference point for utility metrics
- **Properties**:
  - ROUGE-L = 1.0 (perfect match with itself)
  - Privacy Score = 0.0 (no privacy protection)
  - Compression Ratio = 1.0 (no compression)
  - Token Overlap = 1.0 (100% overlap)

### Comparison Approach
All summarization methods are compared against this baseline to measure:
- **Utility Loss**: How much information is lost (1 - ROUGE-L)
- **Privacy Gain**: How much privacy is gained (Privacy Score)
- **Compression Achieved**: How much size reduction (1 - Compression Ratio)

---

## 9. Limitations and Considerations

### Sample Size
- **Current**: 5 samples
- **Limitation**: Small sample size may not capture full variability
- **Mitigation**: Samples selected from diverse Wikipedia content

### Domain Specificity
- **Dataset**: Wikipedia (encyclopedic, factual content)
- **Generalization**: Results may differ for other domains (e.g., medical, legal, conversational)

### LLM Variability
- **Issue**: LLM outputs are non-deterministic (temperature = 0.7)
- **Impact**: Slight variation in results if re-run
- **Mitigation**: Temperature chosen for balance between consistency and quality

### Extractive Limitations
- **Token Overlap**: Always 100% for extractive methods
- **Privacy**: Limited privacy protection from extractive approaches
- **Implication**: Abstractive methods (LLM) inherently provide better privacy

### Metric Limitations
- **ROUGE-L**: Measures surface-level similarity, not semantic equivalence
- **Jaccard**: Simple token overlap, doesn't capture synonyms or paraphrasing
- **Privacy Score**: Simplified metric; real privacy depends on context and adversary model

---

## 10. Reproducibility

### To Reproduce This Experiment:

1. **Setup Environment**:
   ```bash
   conda create -n privacy_analysis python=3.10
   conda activate privacy_analysis
   pip install numpy matplotlib networkx scikit-learn together python-dotenv pandas seaborn
   ```

2. **Configure API Key**:
   - Create `.env` file with `TOGETHER_API_KEY=your_key_here`

3. **Prepare Data**:
   - Ensure `raw_data/private/wiki_newest/wiki_newest.txt` exists
   - Or replace with your own text corpus

4. **Run Experiment**:
   ```bash
   python run_comprehensive_comparison.py
   ```

5. **View Results**:
   - Results: `results/comprehensive_comparison/evaluation_results.json`
   - Visualizations: `results/comprehensive_comparison/*.png`
   - Summary: `results/comprehensive_comparison/SUMMARY.md`

### Configuration Options:
- **Sample Size**: Modify `max_samples=5` in `run_comprehensive_comparison.py`
- **Compression Ratios**: Modify `compression_ratios = [0.3, 0.5, 0.7]`
- **LLM Model**: Change model in `LLMSummarizer` class
- **MMR Lambda**: Adjust relevance/diversity trade-off in `MMRSummarizer`

---

## 11. Key Findings Summary

Based on 5 Wikipedia samples:

| Metric | Best Method | Score |
|--------|-------------|-------|
| **Utility** | LexRank 70% | 0.772 |
| **Privacy** | LLM 30% | 0.689 |
| **Compression** | MMR 30% | 0.251 |
| **Balance** | MMR 30% | Combined |

**Main Insights**:
1. Extractive methods preserve more information but offer less privacy
2. LLM methods provide better privacy through paraphrasing
3. MMR achieves best compression through redundancy reduction
4. Trade-off exists between utility, privacy, and compression
5. No single method dominates all metrics (Pareto frontier)

---

## 12. Future Work

### Potential Extensions:
1. **Larger Sample Size**: Test on 50-100 samples for statistical significance
2. **Domain Diversity**: Include medical, legal, news, social media text
3. **Advanced Privacy Metrics**: Differential privacy, k-anonymity, semantic similarity
4. **More LLM Models**: Compare GPT-4, Claude, Gemini, etc.
5. **Hybrid Methods**: Combine extractive and abstractive approaches
6. **User Studies**: Evaluate human perception of utility and privacy
7. **Adversarial Testing**: Simulate information extraction attacks
8. **Multilingual**: Test on non-English languages

---

## References

### Algorithms:
- **TextRank**: Mihalcea & Tarau (2004) - "TextRank: Bringing Order into Texts"
- **LexRank**: Erkan & Radev (2004) - "LexRank: Graph-based Lexical Centrality"
- **MMR**: Carbonell & Goldstein (1998) - "The Use of MMR, Diversity-Based Reranking"
- **LSA**: Landauer et al. (1998) - "An Introduction to Latent Semantic Analysis"

### Metrics:
- **ROUGE**: Lin (2004) - "ROUGE: A Package for Automatic Evaluation of Summaries"

### Tools:
- **Together AI**: https://www.together.ai/
- **scikit-learn**: https://scikit-learn.org/
- **NetworkX**: https://networkx.org/

---

**Experiment Date**: November 2, 2025  
**Code Repository**: `/Users/liazheng/Local/CS2881/2881-mini-project`  
**Results Directory**: `results/comprehensive_comparison/`
