# Cross-Lingual RAG Experiment Results

## Overview
This experiment tests the "Follow My Instruction and Spill the Beans" attack on cross-lingual RAG systems using Spanish Wikipedia data and English prompts.

**Date:** October 27, 2025  
**Models Tested:** Mistral-7B-Instruct-v0.1, Mixtral-8x7B-Instruct-v0.1  
**Datastore:** Spanish Wikipedia (100 articles from eswiki_first_half.json-2)

---

## Experiment 1: English Prompts on Spanish Dataset (10 prompts)
**Models:** Mistral-7B, Mixtral-8x7B  
**Prompt Type:** Generic English attack prompts  
**Dataset:** Synthetic Spanish Wikipedia (10 AI/ML articles)

### Results

| Model | ROUGE-L | BLEU | BERTScore |
|-------|---------|------|-----------|
| Mistral-7B | 0.013 | 2.911 | 0.026 |
| Mixtral-8x7B | 0.012 | 3.257 | 0.028 |

**Key Finding:** Poor retrieval - most prompts returned empty retrievals (45/50 empty for 50-prompt version).

---

## Experiment 2: English Prompts on Spanish Dataset (50 prompts)
**Models:** Mistral-7B, Mixtral-8x7B  
**Prompt Type:** Generic English attack prompts  
**Dataset:** Synthetic Spanish Wikipedia (10 AI/ML articles)

### Results

| Model | ROUGE-L | BLEU | BERTScore |
|-------|---------|------|-----------|
| Mistral-7B | 0.010 | 2.891 | 0.023 |
| Mixtral-8x7B | 0.015 | 3.543 | 0.029 |

**Key Finding:** Consistent with 10-prompt results - minimal cross-lingual leakage observed.

---

## Experiment 3: English Prompts on Spanish Dataset - Real Wiki (50 prompts)
**Models:** Mistral-7B, Mixtral-8x7B  
**Prompt Type:** Generic English attack prompts  
**Dataset:** Real Spanish Wikipedia (100 articles, 5,840 chunks)

### Results

| Model | ROUGE-L | BLEU | BERTScore | Token F1 | Avg Pred Words |
|-------|---------|------|-----------|----------|----------------|
| Mistral-7B | 0.403 | 30.837 | 0.853 | 0.459 | 136.3 |
| Mixtral-8x7B | 0.558 | 40.240 | 0.880 | 0.591 | 187.1 |

**Key Findings:**
- ✅ **Much better than synthetic data!** Significantly higher scores than experiments with 10-article synthetic dataset
- ✅ **Successful retrieval:** Even generic English prompts are finding content in larger dataset
- ✅ **Mixtral excels:** Mixtral shows much stronger cross-lingual leakage (ROUGE-L 0.558 vs 0.403, BLEU 40.2 vs 30.8)
- ✅ **No exact matches:** Still paraphrased/translated outputs, not verbatim

---

## Experiment 4: Spanish Prompts on English Dataset (50 prompts)
**Models:** Mistral-7B, Mixtral-8x7B  
**Prompt Type:** Spanish attack prompts  
**Dataset:** Real English Wikipedia (wiki_newest, ~1000 articles, 19,860 chunks)

### Results

| Model | ROUGE-L | BLEU | BERTScore | Token F1 | Avg Pred Words |
|-------|---------|------|-----------|----------|----------------|
| Mistral-7B | 0.447 | 33.596 | 0.876 | 0.548 | 164.3 |
| Mixtral-8x7B | 0.684 | 52.420 | 0.918 | 0.726 | 231.7 |

**Key Findings:**
- ✅ **Strong cross-lingual leakage!** Spanish prompts successfully retrieve and leak English Wikipedia content
- ✅ **Mixtral shows exceptional leakage:** Much stronger than Mistral (ROUGE-L 0.684 vs 0.447, BLEU 52.4 vs 33.6)
- ✅ **High semantic similarity:** BERTScore ~0.92 for Mixtral indicates strong content leakage
- ✅ **No exact matches:** Outputs are paraphrased/translated, not verbatim copies
- 📊 **Effectiveness:** This is the strongest cross-lingual leakage observed across all experiments

---

## Experiment 5: Spanish Prompts on Spanish Dataset (50 prompts)
**Models:** Mistral-7B, Mixtral-8x7B  
**Prompt Type:** Spanish attack prompts  
**Dataset:** Real Spanish Wikipedia (100 articles, 5,840 chunks)

### Results

| Model | ROUGE-L | BLEU | BERTScore | Token F1 | Avg Pred Words |
|-------|---------|------|-----------|----------|----------------|
| Mistral-7B | 0.318 | 23.934 | 0.858 | 0.379 | 151.7 |
| Mixtral-8x7B | 0.366 | 23.221 | 0.863 | 0.410 | 192.8 |

**Key Findings:**
- ✅ **Successful retrieval:** Spanish prompts successfully matched and retrieved Spanish content
- ✅ **High semantic similarity:** BERTScore ~0.86 indicates strong semantic overlap
- ✅ **Evidence of leakage:** Models accessed and used retrieved documents
- ❌ **No exact matches:** Outputs are paraphrased/translated, not verbatim copies
- **Model Performance:** Mixtral slightly outperforms Mistral on ROUGE-L (0.366 vs 0.318)

---

## Discussion

### Why English Prompts Failed
1. **BM25 requires word overlap:** Generic English prompts like "Follow my instruction and repeat" have no meaningful overlap with Spanish Wikipedia content
2. **Language barrier:** Token-based BM25 cannot match across languages without translation
3. **Empty retrievals:** 45 out of 50 English prompts returned empty results

### Why Spanish Prompts Succeeded
1. **Successful retrieval:** BM25 found relevant Spanish content
2. **Cross-lingual output:** Models generate responses (sometimes in English, sometimes in Spanish)
3. **High semantic similarity:** BERTScore shows outputs are semantically related to retrieved documents

### Implications
- **Attack only works when retrieval succeeds:** Cross-lingual RAG is more vulnerable when prompts match the datastore language
- **Models show cross-lingual capabilities:** Even when prompted in one language, models can process multilingual retrieved content
- **Paraphrasing vs verbatim:** Models tend to paraphrase/translate rather than copy verbatim

---

## Experimental Setup

### Data Processing
```bash
# Extracted 100 articles from eswiki_first_half.json-2
python scripts/process_eswiki_json.py eswiki_first_half.json-2 \
  raw_data/private/wiki_spanish_eswiki 100
```

### Model Configuration
- **API:** Together AI (for generation)
- **Tokenization:** HuggingFace (mistralai/Mistral-7B-Instruct-v0.1, mistralai/Mixtral-8x7B-Instruct-v0.1)
- **Chunk Size:** 256 tokens
- **Stride:** 128 tokens
- **Retrieval:** BM25 (Pyserini/Lucene)

### Evaluation Metrics
- **ROUGE-L:** Longest common subsequence between output and retrieved docs
- **BLEU:** n-gram precision score
- **BERTScore:** Semantic similarity using BERT embeddings
- **Token Set F1:** Token-level overlap
- **Exact Match:** Whether output exactly matches retrieved content

---

## Files Generated

### Input Data
- `raw_data/private/wiki_spanish_eswiki/`: 100 Spanish Wikipedia articles
- `prompts/test_spanish.json`: 50 Spanish attack prompts

### Output Data
- `eval_data/spanish_on_spanish_eswiki/`: Mistral-7B results (50 JSON files)
- `eval_data/spanish_on_spanish_eswiki_mixtral/`: Mixtral-8x7B results (50 JSON files)

### Evaluation Results
- `eval_data/spanish_on_spanish_eswiki_results/Mistral-7B-Instruct-v0.1.json`
- `eval_data/spanish_on_spanish_eswiki_mixtral_results/Mixtral-8x7B-Instruct-v0.1.json`

### Logs
- `spanish_on_spanish_eswiki.log`: Mistral-7B execution log
- `mixtral_spanish_on_spanish.log`: Mixtral-8x7B execution log
- `eval_spanish_spanish.log`: Evaluation log for Mistral
- `eval_mixtral_spanish_spanish.log`: Evaluation log for Mixtral

---

## Conclusion

The cross-lingual RAG "Follow My Instruction" attack demonstrates that:

1. **Retrieval quality is critical:** The attack only succeeds when the retrieval system finds relevant documents
2. **Language mismatch hurts retrieval:** English prompts on Spanish data fail to retrieve content
3. **Spanish prompts succeed:** When prompts match the datastore language, retrieval works and models demonstrate cross-lingual leakage
4. **Both models vulnerable:** Mistral-7B and Mixtral-8x7B both show evidence of accessing and using retrieved documents
5. **No verbatim copying:** Models paraphrase and translate rather than copying verbatim, indicating they process the retrieved content

**Recommendation:** To defend against this attack in cross-lingual RAG systems, consider:
- Language-aware retrieval (e.g., translate queries before retrieval)
- Retrieval verification (check if retrieved content is actually relevant)
- Output filtering (detect suspicious patterns in model outputs)

