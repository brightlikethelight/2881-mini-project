import sys
sys.path.append(".")

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'False'

import json
import subprocess
import numpy as np
import torch 
from tqdm import tqdm
from typing import List, Dict, Union
from pyserini.search.lucene import LuceneSearcher

from utils.helpers import read_raw_data_dir

# Extension 2: Dense and Hybrid retrieval
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    DENSE_AVAILABLE = True
except ImportError:
    DENSE_AVAILABLE = False
    print("Warning: sentence-transformers or faiss not installed. DenseIndex and HybridIndex will not work.")


class Index(object):
    def __init__(self, raw_data_dir, datastore_dir) -> None:
        assert os.path.exists(raw_data_dir)
        self.raw_data_dir = raw_data_dir
        self.datastore_dir = datastore_dir
    
    def find_most_relevant_k_documents(query: str, k: int):
        raise NotImplementedError


class BM25Index(Index):
    def __init__(self, tokenizer, max_retrieval_seq_length: int, stride: int,
                 raw_data_dir, datastore_dir, recursive=True) -> None:
        super().__init__(raw_data_dir, datastore_dir)
        
        self.tokenizer = tokenizer
        self.max_retrieval_seq_length = max_retrieval_seq_length
        self.stride = stride
        
        if (not os.path.exists(datastore_dir)) or (len(os.listdir(datastore_dir)) == 0):
            os.makedirs(datastore_dir, exist_ok=True)
            
            #! step 1: tokenize raw data 
            print("==> Reading and tokenizing raw data...")
            data = read_raw_data_dir(raw_data_dir=raw_data_dir, recursive=recursive)
            # todo: process very long text?
            all_text = " ".join(data)
            
            all_words = all_text.split()
            step_size = 1024
            chunks_to_tokenize = [all_words[i:i + step_size] for i in range(0, len(all_words), step_size)]
            chunks_to_tokenize = [" ".join(chunk) for chunk in chunks_to_tokenize]
            
            final_tokens = []
            for chunk in tqdm(chunks_to_tokenize):
                tokenizer.parallelism = 8
                tokenized_data = tokenizer(chunk)['input_ids']
                final_tokens.extend(tokenized_data)
            final_tokens = np.array(final_tokens)
            print(f"==> Number of tokens: {len(final_tokens)}.")
            
            #! step 2: split tokenized data into chunks
            print("==> Making chunks...")
            tokens_as_chunks = self._get_token_chunks(
                final_tokens, 
                pad_token=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
            print(f"==> {len(tokens_as_chunks)} chunks in total.")

            self.tokens_dir = os.path.join(datastore_dir, "tokens")
            os.makedirs(self.tokens_dir, exist_ok=True)
            with open(os.path.join(self.tokens_dir, "data.jsonl"), "w") as f:
                for chunk_id, token_chunk in enumerate(tokens_as_chunks):
                    assert len(token_chunk) <= max_retrieval_seq_length
                    text = tokenizer.decode(token_chunk)
                    f.write(json.dumps({
                        "id": str(chunk_id),
                        "contents": text,
                        "input_ids": token_chunk.tolist()
                    })+"\n")
        
            #! step 3: build index on the datastore
            print("==> Start building index for %s at %s" % (self.tokens_dir, datastore_dir))
            command = """python -m pyserini.index.lucene \
            --collection JsonCollection \
            --input '%s' \
            --index '%s' \
            --generator DefaultLuceneDocumentGenerator \
            --storeRaw --threads 1""" % (self.tokens_dir, datastore_dir)
            ret_code = subprocess.run([command],
                                      shell=True,
                                      # stdout=subprocess.DEVNULL,
                                      # stderr=subprocess.STDOUT
                                      )
            if ret_code.returncode != 0:
                print("Failed to build the index")
                exit()
            else:
                print("Successfully built the index")
        else:
            print("==> Datastore exists at: ", datastore_dir)
        
        self.searcher = LuceneSearcher(datastore_dir)
    
    def _get_token_chunks(self, tokens: np.ndarray, pad_token: int) -> np.ndarray:
        assert tokens.ndim == 1, "Tokens should be flattened first!"
        num_tokens = len(tokens)
        tokens_as_chunks = []
        
        for begin_loc in range(0, num_tokens, self.stride):
            end_loc = min(begin_loc + self.max_retrieval_seq_length, num_tokens)
            token_chunk = tokens[begin_loc:end_loc].copy()
        
            if end_loc == num_tokens and len(token_chunk) < self.max_retrieval_seq_length:
                pads = np.array([pad_token for _ in range(self.max_retrieval_seq_length - len(token_chunk))])
                token_chunk = np.concatenate([token_chunk, pads])
        
            assert len(token_chunk) == self.max_retrieval_seq_length
            
            tokens_as_chunks.append(token_chunk)
        
        tokens_as_chunks = np.stack(tokens_as_chunks)
        return tokens_as_chunks

    def find_most_relevant_k_documents(self, query: str, k: int) -> List[str]:
        hits = self.searcher.search(query, k=k)
        docs = []
        for hit in hits:
            docid = hit.docid
            raw = self.searcher.doc(docid).raw()
            input_ids = json.loads(raw)["input_ids"]
            doc_str = self.tokenizer.decode(input_ids)
            docs.append(doc_str)
        return docs


class DenseIndex(Index):
    """
    Extension 2: Dense retrieval using sentence-transformers + FAISS.
    """
    def __init__(self, tokenizer, dense_model: str, faiss_index_type: str,
                 max_retrieval_seq_length: int, stride: int,
                 raw_data_dir, datastore_dir, recursive=True) -> None:
        super().__init__(raw_data_dir, datastore_dir)

        if not DENSE_AVAILABLE:
            raise ImportError("sentence-transformers and faiss required for DenseIndex. "
                            "Install: pip install sentence-transformers faiss-cpu")

        self.tokenizer = tokenizer
        self.dense_model_name = dense_model
        self.faiss_index_type = faiss_index_type
        self.max_retrieval_seq_length = max_retrieval_seq_length
        self.stride = stride

        # Load dense encoder
        print(f"==> Loading dense encoder: {dense_model}")
        self.encoder = SentenceTransformer(dense_model)

        # Check if index already exists
        chunks_file = os.path.join(datastore_dir, "chunks.jsonl")
        faiss_index_file = os.path.join(datastore_dir, "dense.index")

        if os.path.exists(chunks_file) and os.path.exists(faiss_index_file):
            print(f"==> Loading existing dense index from {datastore_dir}")
            self._load_index(chunks_file, faiss_index_file)
        else:
            print(f"==> Building dense index at {datastore_dir}")
            os.makedirs(datastore_dir, exist_ok=True)
            self._build_index(raw_data_dir, datastore_dir, recursive, chunks_file, faiss_index_file)

    def _build_index(self, raw_data_dir, datastore_dir, recursive, chunks_file, faiss_index_file):
        """Build dense index from raw data."""
        # Step 1: Read and tokenize raw data
        print("==> Reading and tokenizing raw data...")
        data = read_raw_data_dir(raw_data_dir=raw_data_dir, recursive=recursive)
        all_text = " ".join(data)

        all_words = all_text.split()
        step_size = 1024
        chunks_to_tokenize = [all_words[i:i + step_size] for i in range(0, len(all_words), step_size)]
        chunks_to_tokenize = [" ".join(chunk) for chunk in chunks_to_tokenize]

        final_tokens = []
        for chunk in tqdm(chunks_to_tokenize, desc="Tokenizing"):
            tokenized_data = self.tokenizer(chunk)['input_ids']
            final_tokens.extend(tokenized_data)
        final_tokens = np.array(final_tokens)
        print(f"==> Number of tokens: {len(final_tokens)}.")

        # Step 2: Split into chunks
        print("==> Making chunks...")
        pad_token = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        tokens_as_chunks = self._get_token_chunks(final_tokens, pad_token)
        print(f"==> {len(tokens_as_chunks)} chunks in total.")

        # Step 3: Decode chunks to text
        self.chunks = []
        self.chunk_ids = []
        with open(chunks_file, "w") as f:
            for chunk_id, token_chunk in enumerate(tqdm(tokens_as_chunks, desc="Decoding chunks")):
                text = self.tokenizer.decode(token_chunk, skip_special_tokens=True).strip()
                if text:  # Skip empty chunks
                    self.chunks.append(text)
                    self.chunk_ids.append(chunk_id)
                    f.write(json.dumps({"id": chunk_id, "text": text}) + "\n")

        print(f"==> {len(self.chunks)} non-empty chunks.")

        # Step 4: Encode chunks with dense model
        print("==> Encoding chunks with dense model...")
        self.embeddings = self.encoder.encode(self.chunks, show_progress_bar=True, batch_size=32)
        self.embeddings = np.array(self.embeddings).astype('float32')
        print(f"==> Embeddings shape: {self.embeddings.shape}")

        # Step 5: Build FAISS index
        print(f"==> Building FAISS index ({self.faiss_index_type})...")
        dimension = self.embeddings.shape[1]

        if self.faiss_index_type == 'Flat':
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        elif self.faiss_index_type == 'IVFFlat':
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, min(100, len(self.chunks)))
            self.index.train(self.embeddings)
        else:
            raise ValueError(f"Unknown FAISS index type: {self.faiss_index_type}")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

        # Save index
        faiss.write_index(self.index, faiss_index_file)
        print(f"==> FAISS index saved to {faiss_index_file}")

    def _load_index(self, chunks_file, faiss_index_file):
        """Load pre-built dense index."""
        # Load chunks
        self.chunks = []
        self.chunk_ids = []
        with open(chunks_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.chunks.append(data['text'])
                self.chunk_ids.append(data['id'])

        # Load FAISS index
        self.index = faiss.read_index(faiss_index_file)
        print(f"==> Loaded {len(self.chunks)} chunks and FAISS index with {self.index.ntotal} vectors")

    def _get_token_chunks(self, tokens: np.ndarray, pad_token: int) -> np.ndarray:
        """Same chunking logic as BM25Index."""
        assert tokens.ndim == 1, "Tokens should be flattened first!"
        num_tokens = len(tokens)
        tokens_as_chunks = []

        for begin_loc in range(0, num_tokens, self.stride):
            end_loc = min(begin_loc + self.max_retrieval_seq_length, num_tokens)
            token_chunk = tokens[begin_loc:end_loc].copy()

            if end_loc == num_tokens and len(token_chunk) < self.max_retrieval_seq_length:
                pads = np.array([pad_token for _ in range(self.max_retrieval_seq_length - len(token_chunk))])
                token_chunk = np.concatenate([token_chunk, pads])

            assert len(token_chunk) == self.max_retrieval_seq_length
            tokens_as_chunks.append(token_chunk)

        tokens_as_chunks = np.stack(tokens_as_chunks)
        return tokens_as_chunks

    def find_most_relevant_k_documents(self, query: str, k: int) -> List[str]:
        """Find top-k documents using dense retrieval."""
        # Encode query
        query_embedding = self.encoder.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, k)

        # Return top-k documents
        docs = []
        for idx in indices[0]:
            if idx >= 0 and idx < len(self.chunks):
                docs.append(self.chunks[idx])

        return docs


class HybridIndex(Index):
    """
    Extension 2: Hybrid retrieval combining BM25 and Dense retrieval.
    Uses reciprocal rank fusion (RRF) to merge rankings.
    """
    def __init__(self, tokenizer, dense_model: str, faiss_index_type: str,
                 hybrid_alpha: float, max_retrieval_seq_length: int, stride: int,
                 raw_data_dir, datastore_dir, recursive=True) -> None:
        super().__init__(raw_data_dir, datastore_dir)

        self.hybrid_alpha = hybrid_alpha  # 0.5 = equal weight to BM25 and dense

        # Create subdirectories for each index type
        bm25_dir = os.path.join(datastore_dir, "bm25")
        dense_dir = os.path.join(datastore_dir, "dense")

        print("==> Initializing hybrid index (BM25 + Dense)")

        # Initialize BM25 index
        print("==> Building BM25 component...")
        self.bm25_index = BM25Index(
            tokenizer=tokenizer,
            max_retrieval_seq_length=max_retrieval_seq_length,
            stride=stride,
            raw_data_dir=raw_data_dir,
            datastore_dir=bm25_dir,
            recursive=recursive
        )

        # Initialize Dense index
        print("==> Building Dense component...")
        self.dense_index = DenseIndex(
            tokenizer=tokenizer,
            dense_model=dense_model,
            faiss_index_type=faiss_index_type,
            max_retrieval_seq_length=max_retrieval_seq_length,
            stride=stride,
            raw_data_dir=raw_data_dir,
            datastore_dir=dense_dir,
            recursive=recursive
        )

    def find_most_relevant_k_documents(self, query: str, k: int) -> List[str]:
        """
        Find top-k documents using hybrid retrieval with reciprocal rank fusion.

        RRF score for document d: sum_r (1 / (rank_r(d) + 60))
        where rank_r(d) is the rank of document d in retrieval system r.
        """
        # Get top-2k from each retriever
        k_fetch = k * 2

        bm25_docs = self.bm25_index.find_most_relevant_k_documents(query, k_fetch)
        dense_docs = self.dense_index.find_most_relevant_k_documents(query, k_fetch)

        # Compute RRF scores
        rrf_scores = {}
        k_rrf = 60  # RRF constant

        # BM25 rankings
        for rank, doc in enumerate(bm25_docs):
            if doc not in rrf_scores:
                rrf_scores[doc] = 0
            rrf_scores[doc] += self.hybrid_alpha / (rank + k_rrf)

        # Dense rankings
        for rank, doc in enumerate(dense_docs):
            if doc not in rrf_scores:
                rrf_scores[doc] = 0
            rrf_scores[doc] += (1 - self.hybrid_alpha) / (rank + k_rrf)

        # Sort by RRF score and return top-k
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_docs = [doc for doc, score in sorted_docs[:k]]

        return top_k_docs


if __name__ == '__main__':
    pass