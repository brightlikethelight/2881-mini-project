"""
Comprehensive Summarization Methods Comparison
Compare TextRank, LexRank, SumBasic, LSA/SVD, MMR, and LLM at different compression ratios
"""

import json
import re
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import networkx as nx
from together import Together
from dotenv import load_dotenv
import asyncio
import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

load_dotenv()

# Tokenization utilities
def simple_tokenize(text):
    if not text:
        return []
    return re.findall(r'\b\w+\b', text.lower())

def sentence_tokenize(text):
    sentences = re.split(r'[.!?]+\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# Metrics
def calculate_rouge_l(candidate, reference):
    def lcs_length(X, Y):
        m, n = len(X), len(Y)
        L = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i-1] == Y[j-1]:
                    L[i][j] = L[i-1][j-1] + 1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])
        return L[m][n]
    
    candidate_tokens = simple_tokenize(candidate)
    reference_tokens = simple_tokenize(reference)
    
    if not candidate_tokens or not reference_tokens:
        return 0.0
    
    lcs = lcs_length(candidate_tokens, reference_tokens)
    precision = lcs / len(candidate_tokens) if candidate_tokens else 0
    recall = lcs / len(reference_tokens) if reference_tokens else 0
    
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    return 0.0

def calculate_jaccard_similarity(text1, text2):
    set1 = set(simple_tokenize(text1))
    set2 = set(simple_tokenize(text2))
    if not set1 and not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def calculate_compression_ratio(original, summary):
    orig_len = len(simple_tokenize(original))
    summ_len = len(simple_tokenize(summary))
    return summ_len / orig_len if orig_len > 0 else 0.0

def calculate_token_overlap(summary, original):
    summary_tokens = set(simple_tokenize(summary))
    original_tokens = set(simple_tokenize(original))
    if not summary_tokens:
        return 0.0
    overlap = len(summary_tokens & original_tokens)
    return overlap / len(summary_tokens)

# Summarization Methods
class TextRankSummarizer:
    def __init__(self, compression_ratio=0.3):
        self.compression_ratio = compression_ratio
    
    def _build_similarity_matrix(self, sentences):
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    continue
                words_i = set(simple_tokenize(sentences[i]))
                words_j = set(simple_tokenize(sentences[j]))
                if not words_i or not words_j:
                    continue
                intersection = len(words_i & words_j)
                union = len(words_i | words_j)
                if union > 0:
                    similarity = intersection / union
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity
        return similarity_matrix
    
    def summarize(self, text):
        sentences = sentence_tokenize(text)
        if len(sentences) <= 1:
            return text
        similarity_matrix = self._build_similarity_matrix(sentences)
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph, alpha=0.85)
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        num_sentences = max(1, int(len(sentences) * self.compression_ratio))
        selected = ranked_sentences[:num_sentences]
        selected_indices = [sentences.index(s) for _, s in selected]
        selected_indices.sort()
        return ' '.join([sentences[i] for i in selected_indices])

class LexRankSummarizer:
    def __init__(self, compression_ratio=0.3):
        self.compression_ratio = compression_ratio
    
    def _build_cosine_matrix(self, sentences):
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            similarity_matrix = sklearn_cosine_similarity(tfidf_matrix)
            return similarity_matrix
        except:
            return np.zeros((len(sentences), len(sentences)))
    
    def summarize(self, text):
        sentences = sentence_tokenize(text)
        if len(sentences) <= 1:
            return text
        similarity_matrix = self._build_cosine_matrix(sentences)
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph, alpha=0.85)
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        num_sentences = max(1, int(len(sentences) * self.compression_ratio))
        selected = ranked_sentences[:num_sentences]
        selected_indices = [sentences.index(s) for _, s in selected]
        selected_indices.sort()
        return ' '.join([sentences[i] for i in selected_indices])

class SumBasicSummarizer:
    def __init__(self, compression_ratio=0.3):
        self.compression_ratio = compression_ratio
    
    def summarize(self, text):
        sentences = sentence_tokenize(text)
        if len(sentences) <= 1:
            return text
        
        # Calculate word frequencies
        all_words = []
        for sent in sentences:
            all_words.extend(simple_tokenize(sent))
        
        word_freq = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Normalize frequencies
        max_freq = max(word_freq.values()) if word_freq else 1
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
        
        # Score sentences
        sentence_scores = []
        for sent in sentences:
            words = simple_tokenize(sent)
            if words:
                score = sum(word_freq.get(word, 0) for word in words) / len(words)
                sentence_scores.append(score)
            else:
                sentence_scores.append(0)
        
        # Select top sentences
        num_sentences = max(1, int(len(sentences) * self.compression_ratio))
        top_indices = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:num_sentences]
        top_indices.sort()
        
        return ' '.join([sentences[i] for i in top_indices])

class LSASummarizer:
    def __init__(self, compression_ratio=0.3):
        self.compression_ratio = compression_ratio
    
    def summarize(self, text):
        sentences = sentence_tokenize(text)
        if len(sentences) <= 1:
            return text
        
        try:
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(sentences)
            
            n_components = min(10, len(sentences), X.shape[1])
            svd = TruncatedSVD(n_components=n_components)
            svd.fit(X)
            
            # Score sentences based on their representation in top singular vectors
            sentence_scores = []
            for i in range(len(sentences)):
                score = sum(svd.components_[j][i] ** 2 for j in range(min(3, n_components)))
                sentence_scores.append(score)
            
            num_sentences = max(1, int(len(sentences) * self.compression_ratio))
            top_indices = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:num_sentences]
            top_indices.sort()
            
            return ' '.join([sentences[i] for i in top_indices])
        except:
            return sentences[0] if sentences else text

class MMRSummarizer:
    def __init__(self, compression_ratio=0.3, lambda_param=0.7):
        self.compression_ratio = compression_ratio
        self.lambda_param = lambda_param
    
    def summarize(self, text):
        sentences = sentence_tokenize(text)
        if len(sentences) <= 1:
            return text
        
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate similarity to document (mean of all sentences)
            doc_vector = tfidf_matrix.mean(axis=0)
            sim_to_doc = sklearn_cosine_similarity(tfidf_matrix, doc_vector).flatten()
            
            # MMR selection
            selected_indices = []
            num_sentences = max(1, int(len(sentences) * self.compression_ratio))
            
            for _ in range(num_sentences):
                mmr_scores = []
                for i in range(len(sentences)):
                    if i in selected_indices:
                        mmr_scores.append(-float('inf'))
                        continue
                    
                    relevance = sim_to_doc[i]
                    
                    if selected_indices:
                        max_sim = max(sklearn_cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[j:j+1])[0][0] 
                                    for j in selected_indices)
                        redundancy = max_sim
                    else:
                        redundancy = 0
                    
                    mmr = self.lambda_param * relevance - (1 - self.lambda_param) * redundancy
                    mmr_scores.append(mmr)
                
                best_idx = max(range(len(mmr_scores)), key=lambda i: mmr_scores[i])
                selected_indices.append(best_idx)
            
            selected_indices.sort()
            return ' '.join([sentences[i] for i in selected_indices])
        except:
            return sentences[0] if sentences else text

class LLMSummarizer:
    def __init__(self, api_key=None, model="meta-llama/Llama-3.2-3B-Instruct-Turbo", compression_ratio=0.3):
        self.api_key = api_key or os.getenv('TOGETHER_API_KEY')
        self.client = Together(api_key=self.api_key)
        self.model = model
        self.compression_ratio = compression_ratio
    
    async def summarize_async(self, text):
        original_words = len(simple_tokenize(text))
        target_words = int(original_words * self.compression_ratio)
        
        prompt = f"""Summarize the following text in approximately {target_words} words. Keep the most important information and maintain clarity.

Text:
{text}

Summary:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=min(target_words * 2, 1024),
                temperature=0.3,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  Error in LLM summarization: {e}")
            return ""

def load_wiki_data(file_path, max_samples=10):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 200]
    samples = []
    for i, para in enumerate(paragraphs[:max_samples]):
        samples.append({'id': f'wiki_{i}', 'text': para})
    return samples

async def generate_summaries_async(samples, compression_ratios):
    methods = {
        'no_compression': (None, None),
    }
    
    # Add extractive methods
    for ratio in compression_ratios:
        methods[f'textrank_{int(ratio*100)}%'] = (TextRankSummarizer(ratio), 'extractive')
        methods[f'lexrank_{int(ratio*100)}%'] = (LexRankSummarizer(ratio), 'extractive')
        methods[f'sumbasic_{int(ratio*100)}%'] = (SumBasicSummarizer(ratio), 'extractive')
        methods[f'lsa_{int(ratio*100)}%'] = (LSASummarizer(ratio), 'extractive')
        methods[f'mmr_{int(ratio*100)}%'] = (MMRSummarizer(ratio), 'extractive')
    
    # Add LLM methods
    for ratio in compression_ratios:
        methods[f'llm_{int(ratio*100)}%'] = (LLMSummarizer(compression_ratio=ratio), 'abstractive')
    
    results = []
    
    for sample in samples:
        print(f"\nProcessing {sample['id']}...")
        sample_results = {
            'id': sample['id'],
            'original': sample['text'],
            'summaries': {}
        }
        
        for method_name, (summarizer, method_type) in methods.items():
            try:
                if method_name == 'no_compression':
                    summary = sample['text']
                elif method_type == 'abstractive':
                    print(f"  Calling LLM for {method_name}...")
                    summary = await summarizer.summarize_async(sample['text'])
                    time.sleep(0.5)
                else:
                    summary = summarizer.summarize(sample['text'])
                
                sample_results['summaries'][method_name] = summary
                word_count = len(simple_tokenize(summary))
                print(f"  ✓ {method_name}: {word_count} words")
            except Exception as e:
                print(f"  ✗ Error with {method_name}: {e}")
                sample_results['summaries'][method_name] = ""
        
        results.append(sample_results)
    
    return results

def evaluate_methods(results, reference_key='no_compression'):
    evaluation = {}
    
    for method_name in results[0]['summaries'].keys():
        evaluation[method_name] = {
            'rouge_l': [],
            'jaccard_similarity': [],
            'token_overlap': [],
            'compression_ratio': [],
            'privacy_score': []
        }
    
    for result in results:
        original = result['original']
        reference = result['summaries'][reference_key]
        
        for method_name, summary in result['summaries'].items():
            if not summary:
                continue
            
            rouge_l = calculate_rouge_l(summary, reference)
            jaccard = calculate_jaccard_similarity(summary, original)
            overlap = calculate_token_overlap(summary, original)
            compression = calculate_compression_ratio(original, summary)
            privacy = 1 - jaccard
            
            evaluation[method_name]['rouge_l'].append(rouge_l)
            evaluation[method_name]['jaccard_similarity'].append(jaccard)
            evaluation[method_name]['token_overlap'].append(overlap)
            evaluation[method_name]['compression_ratio'].append(compression)
            evaluation[method_name]['privacy_score'].append(privacy)
    
    avg_evaluation = {}
    for method_name, metrics in evaluation.items():
        avg_evaluation[method_name] = {
            metric: np.mean(values) if values else 0.0
            for metric, values in metrics.items()
        }
    
    return avg_evaluation

# Continued in next file due to length...
