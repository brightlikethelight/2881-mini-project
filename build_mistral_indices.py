#!/usr/bin/env python3
"""
Build BM25 indices for Mistral models to avoid index building during experiments.
This ensures experiments start immediately without waiting for index construction.
"""

from modules.Index import BM25Index
from transformers import AutoTokenizer
import sys

def build_index(model_name, chunk_size, stride=128):
    """Build BM25 index for a specific model and chunk size."""
    print(f"\n{'='*80}")
    print(f"Building BM25 index for: {model_name}")
    print(f"Chunk size: {chunk_size}, Stride: {stride}")
    print(f"{'='*80}\n")
    
    try:
        # Load tokenizer for this specific model (use_fast=False for Mistral compatibility)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        # Build index
        datastore_path = f"datastore/RIC_LM+wiki_newest+{model_name.split('/')[-1]}+{chunk_size}+{stride}"
        idx = BM25Index(
            tokenizer=tokenizer,
            max_retrieval_seq_length=chunk_size,
            stride=stride,
            raw_data_dir="raw_data/private/wiki_newest",
            datastore_dir=datastore_path,
            recursive=True
        )
        
        print(f"✓ Successfully built index at: {datastore_path}\n")
        return True
        
    except Exception as e:
        print(f"✗ Failed to build index: {e}\n")
        return False

def main():
    """Build all required indices."""
    
    models = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
    ]
    
    # Extension 1 needs: 256 chunks (default)
    # Extension 2 needs: 64, 128, 256 chunks (full grid)
    chunk_sizes = [64, 128, 256]
    
    total = len(models) * len(chunk_sizes)
    success_count = 0
    
    print(f"\n{'='*80}")
    print(f"Building {total} BM25 indices for Mistral experiments")
    print(f"{'='*80}\n")
    
    for model_name in models:
        for chunk_size in chunk_sizes:
            success = build_index(model_name, chunk_size)
            if success:
                success_count += 1
    
    print(f"\n{'='*80}")
    print(f"Index Building Complete!")
    print(f"Success: {success_count}/{total}")
    print(f"Failed: {total - success_count}/{total}")
    print(f"{'='*80}\n")
    
    if success_count == total:
        print("✓ All indices built successfully. Ready to run experiments!")
        return 0
    else:
        print("✗ Some indices failed to build. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
