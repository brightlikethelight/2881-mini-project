# Extension 2: Retriever Surgery

**Research Question:** How do retrieval parameters (chunk size, top-k, retriever type) affect verbatim copying in RAG systems?

## Overview

This extension systematically varies three retrieval parameters:

1. **Chunk Size**: {64, 128, 256} tokens
2. **Top-K**: {2, 4, 8} retrieved documents
3. **Retriever Type**: {BM25, Dense, Hybrid}

**Total: 3 × 3 × 3 = 27 configurations**

## Quick Start

### 1. Install Dependencies

```bash
pip install sentence-transformers faiss-cpu
```

### 2. Local Testing (gpt2, 10 prompts, 3 configs)

```bash
# Run retriever sweep (subset)
python extensions/2_retriever_surgery/scripts/run_retriever_sweep.py \
    --api hf \
    --hf_ckpt gpt2 \
    --num_prompts 10 \
    --configs_subset

# Analyze results
python extensions/2_retriever_surgery/scripts/analyze_positions.py \
    --results_dir extensions/2_retriever_surgery/results
```

### 3. Full Experiments (Llama-2-7B, 50 prompts, all 27 configs)

```bash
# Run retriever sweep via Together API
python extensions/2_retriever_surgery/scripts/run_retriever_sweep.py \
    --api together \
    --hf_ckpt meta-llama/Llama-2-7b-chat-hf \
    --together_ckpt togethercomputer/llama-2-7b-chat \
    --num_prompts 50

# Analyze results
python extensions/2_retriever_surgery/scripts/analyze_positions.py
```

## Configuration Grid

| Config Name | Chunk Size | Top-K | Retriever | Description |
|-------------|-----------|-------|-----------|-------------|
| chunk64_k2_bm25 | 64 | 2 | BM25 | Small chunks, few docs, sparse retrieval |
| chunk64_k2_dense | 64 | 2 | Dense | Small chunks, few docs, dense retrieval |
| chunk64_k2_hybrid | 64 | 2 | Hybrid | Small chunks, few docs, hybrid retrieval |
| ... | ... | ... | ... | ... |
| chunk256_k8_hybrid | 256 | 8 | Hybrid | Large chunks, many docs, hybrid retrieval |

**Representative Subset (3 configs for quick testing):**
- chunk128_k4_bm25 (baseline)
- chunk256_k8_dense (dense + large + many)
- chunk64_k2_hybrid (hybrid + small + few)

## Metrics

- **Utility**: QA F1 (token-set F1)
- **Leakage**: ROUGE-L, Verbatim 5-gram rate
- **Position Bias**: Verbatim copying rate for each document position (1st, 2nd, ..., k-th)

## Hypotheses

1. **Chunk Size**: Larger chunks → more copying (more verbatim text available)
2. **Top-K**: More documents → more copying (more opportunities to copy)
3. **Retriever Type**: Dense < BM25 < Hybrid copying rate (dense retrieval is less exploitable)
4. **Position Bias**: Strong position effects ("lost in the middle" phenomenon)

## Expected Results

**Chunk Size Effect:**
- 64 tokens: Low copying (minimal verbatim text per chunk)
- 128 tokens: Moderate copying (paper's default)
- 256 tokens: High copying (large verbatim chunks)

**Top-K Effect:**
- k=2: Low copying (limited context)
- k=4: Moderate copying (more options)
- k=8: High copying (many opportunities)

**Retriever Type:**
- BM25: Highest copying (precise keyword matching)
- Dense: Lowest copying (semantic matching, less exact)
- Hybrid: Middle ground

**Position Bias:**
- U-shaped curve: Documents at positions 1 and k are copied more than middle positions
- Replicates "lost in the middle" findings from Liu et al. (2023)

## Output Structure

```
extensions/2_retriever_surgery/results/
├── outputs/                           # Raw model outputs
│   ├── chunk64_k2_bm25/
│   │   └── results.json
│   ├── chunk64_k2_dense/
│   │   └── results.json
│   └── ...
├── plots/                             # Analysis plots
│   ├── chunk_size_effect.png
│   ├── top_k_effect.png
│   ├── retriever_comparison.png
│   ├── position_bias.png
│   └── metrics.json
├── metrics_summary.md                 # Summary table
└── summary.json                       # Experiment metadata
```

## Files

- `scripts/run_retriever_sweep.py`: Run grid search over retrieval parameters
- `scripts/analyze_positions.py`: Compute metrics and generate plots
- `configs/`: (Optional) JSON configs for custom parameter combinations

## Implementation Details

### Core Modifications

**modules/Index.py (lines 143-294):**
- Added `DenseIndex` class using sentence-transformers + FAISS
- Encodes chunks with `all-mpnet-base-v2`
- Searches via cosine similarity

**modules/Index.py (lines 297-372):**
- Added `HybridIndex` class combining BM25 + Dense
- Uses Reciprocal Rank Fusion (RRF) to merge rankings
- Configurable α parameter (default 0.5 = equal weight)

**modules/RALM.py (lines 44-79):**
- Updated to support `index_name ∈ {bm25, dense, hybrid}`
- Instantiates appropriate Index class based on config

**utils/argparser.py (lines 49-52):**
```python
# Extension 2: Retriever surgery
dense_model: str = field(default='sentence-transformers/all-mpnet-base-v2')
faiss_index_type: str = field(default='Flat')
hybrid_alpha: float = field(default=0.5)
```

## Retriever Details

### BM25 (Sparse)
- Uses Pyserini/Lucene for indexing
- TF-IDF + BM25 scoring
- Fast, keyword-based matching

### Dense (Neural)
- Sentence-BERT embeddings (`all-mpnet-base-v2`)
- FAISS index (Flat or IVFFlat)
- Semantic similarity via cosine distance
- Slower than BM25, but better semantic matching

### Hybrid
- Combines BM25 and Dense with RRF
- RRF score: `1 / (rank_bm25 + 60)` + `1 / (rank_dense + 60)`
- α parameter balances BM25 vs Dense weight
- Best of both worlds: keyword precision + semantic understanding

## Position Bias Analysis

The `compute_position_bias()` function measures how much each document position contributes to verbatim copying:

```python
def compute_position_bias(results, top_k):
    for each example:
        for position in 1..k:
            compute verbatim_rate(prediction, retrieved_docs[position])
    return [avg_rate_pos_1, avg_rate_pos_2, ..., avg_rate_pos_k]
```

This tests the "lost in the middle" hypothesis: LLMs show recency/primacy bias, copying more from first and last retrieved documents.

## Cost Estimate

**Together API Costs (Llama-2-7B):**
- 27 configs × 50 prompts × ~500 tokens/prompt = ~675K tokens
- Input: 675K tokens × $0.20/1M = $0.135
- Output: 45K tokens × $0.20/1M = $0.009
- **Total: ~$0.15 per full run**

**Note:** Dense index building takes ~5-10 minutes on CPU for 100 Wikipedia articles.

## Notes

- **Dense retrieval requires:**
  - `sentence-transformers` (pip install)
  - `faiss-cpu` (or `faiss-gpu` for CUDA)
- First run will build all indices (BM25, Dense) - takes 10-20 minutes
- Subsequent runs reuse cached indices from `datastore/`
- Apple Silicon Macs: Use `faiss-cpu` (faiss-gpu not available)
- For faster dense encoding, use GPU if available

## Troubleshooting

### Error: `ImportError: sentence-transformers not installed`
```bash
pip install sentence-transformers faiss-cpu
```

### Error: `FAISS: Can't find iostream`
This is a macOS compilation issue. Use pre-built `faiss-cpu`:
```bash
conda install -c pytorch faiss-cpu
```

### Dense indexing is very slow
- Use GPU: `pip install faiss-gpu`
- Or reduce batch size: Edit `modules/Index.py` line 218, change `batch_size=32` to `batch_size=8`

## Next Steps

1. Run full grid search with Llama-2-7B and Mistral-7B
2. Generate all plots (chunk size, top-k, retriever, position bias)
3. Analyze which parameters most strongly affect copying
4. Write up findings comparing with Extension 1 (defense-based mitigation)
5. Propose hybrid approach: optimal retrieval parameters + prompt-level defenses

## References

- Liu et al. (2023): "Lost in the Middle: How Language Models Use Long Contexts"
- Sentence-BERT: https://www.sbert.net/
- FAISS: https://github.com/facebookresearch/faiss
- Reciprocal Rank Fusion: Cormack et al. (2009)
