#!/usr/bin/env python3
"""
Test BM25 datastore creation and retrieval.

Usage:
    python scripts/test_datastore.py \
        --raw_data_dir raw_data/wikipedia_nov2023 \
        --model meta-llama/Llama-2-7b-hf

Quick verification that datastore building and retrieval work correctly.
For ICLR 2025: "Follow My Instruction and Spill the Beans"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.Index import BM25Index
from transformers import AutoTokenizer


def test_datastore(raw_data_dir: str, model: str, verbose: bool = True):
    """Test datastore creation and retrieval."""

    if verbose:
        print(f"{'='*60}")
        print(f"Testing BM25 Datastore")
        print(f"{'='*60}\n")

    # Check if raw data exists
    if not Path(raw_data_dir).exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_data_dir}\n"
            f"Run: python scripts/fetch_wikipedia.py"
        )

    # Count txt files
    txt_files = list(Path(raw_data_dir).glob("*.txt"))
    if not txt_files:
        raise ValueError(
            f"No .txt files found in: {raw_data_dir}\n"
            f"Expected article_*.txt files"
        )

    if verbose:
        print(f"✓ Found {len(txt_files)} article files in {raw_data_dir}")

    # Load tokenizer
    if verbose:
        print(f"\nLoading tokenizer from: {model}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        if verbose:
            print(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}")
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        print(f"\nMake sure you have:")
        print(f"  1. Internet connection (for downloading)")
        print(f"  2. HuggingFace access (if model requires it)")
        print(f"  3. Correct model name: {model}")
        sys.exit(1)

    # Build or load BM25 index
    if verbose:
        print(f"\nBuilding/loading BM25 index...")
        print(f"  Parameters:")
        print(f"    max_retrieval_seq_length: 256")
        print(f"    stride: 128")

    try:
        index = BM25Index(
            tokenizer=tokenizer,
            max_retrieval_seq_length=256,
            stride=128,
            raw_data_dir=raw_data_dir,
            datastore_dir="./datastore/test_wikipedia"
        )
        if verbose:
            print(f"✓ BM25 index ready")
    except Exception as e:
        print(f"\n✗ Error building index: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test retrieval with sample queries
    test_queries = [
        "Tell me about recent events in 2023",
        "What happened in November 2023?",
        "Explain artificial intelligence and machine learning",
        "Describe climate change and global warming",
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing Retrieval (k=3 documents per query)")
        print(f"{'='*60}\n")

    for i, query in enumerate(test_queries, 1):
        if verbose:
            print(f"Query {i}: {query}")

        try:
            docs = index.find_most_relevant_k_documents(query, k=3)

            if verbose:
                print(f"  Retrieved {len(docs)} documents:")
                for j, doc in enumerate(docs, 1):
                    # Show first 100 chars of each doc
                    preview = doc[:100].replace('\n', ' ')
                    print(f"    {j}. {preview}...")
                print()

        except Exception as e:
            print(f"  ✗ Error: {e}")
            sys.exit(1)

    # Success!
    if verbose:
        print(f"{'='*60}")
        print(f"✓ All tests passed!")
        print(f"{'='*60}\n")
        print(f"Datastore is ready for experiments.")
        print(f"\nNext steps:")
        print(f"  1. Generate attack prompts: python scripts/generate_prompts.py")
        print(f"  2. Run inference: python main.py --task io ...")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Test BM25 datastore creation and retrieval',
        epilog='For ICLR 2025 paper: "Follow My Instruction and Spill the Beans"'
    )
    parser.add_argument(
        '--raw_data_dir',
        default='raw_data/wikipedia_nov2023',
        help='Directory with article text files (default: raw_data/wikipedia_nov2023)'
    )
    parser.add_argument(
        '--model',
        default='meta-llama/Llama-2-7b-hf',
        help='Model for tokenizer (default: meta-llama/Llama-2-7b-hf)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    try:
        test_datastore(
            raw_data_dir=args.raw_data_dir,
            model=args.model,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
