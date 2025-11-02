#!/usr/bin/env python3
"""
Extension 2: Retriever Surgery - Retriever Sweep Script

This script runs a grid search over retrieval parameters to measure their effect on copying:
- chunk_size ∈ {64, 128, 256}
- top-k ∈ {2, 4, 8}
- retriever ∈ {bm25, dense, hybrid}
Total: 27 configurations

Usage:
    # Quick test with gpt2 (local, 3 configs)
    python extensions/2_retriever_surgery/scripts/run_retriever_sweep.py \
        --api hf \
        --hf_ckpt gpt2 \
        --num_prompts 10 \
        --configs_subset

    # Full run with Llama-2-7B (Together API, all 27 configs)
    python extensions/2_retriever_surgery/scripts/run_retriever_sweep.py \
        --api together \
        --hf_ckpt meta-llama/Llama-2-7b-chat-hf \
        --together_ckpt togethercomputer/llama-2-7b-chat \
        --num_prompts 50
"""

import os
import sys
import json
import argparse
import subprocess
import itertools
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description="Run retriever sweep for Extension 2")

    # Model selection
    parser.add_argument("--api", type=str, required=True, choices=["hf", "together"],
                        help="API to use: 'hf' for local HuggingFace, 'together' for Together.ai")
    parser.add_argument("--hf_ckpt", type=str, required=True,
                        help="HuggingFace model checkpoint")
    parser.add_argument("--together_ckpt", type=str, default=None,
                        help="Together.ai model checkpoint (required if api=together)")
    parser.add_argument("--is_chat_model", type=str, default="false",
                        help="Whether the model is a chat model (true/false)")

    # Dataset selection
    parser.add_argument("--raw_data_dir", type=str, default="raw_data/private/wiki_newest",
                        help="Directory containing private documents")
    parser.add_argument("--prompts_file", type=str, default="prompts/copy_50.json",
                        help="Adversarial prompts file")
    parser.add_argument("--num_prompts", type=int, default=50,
                        help="Number of prompts to use (for testing)")

    # Grid search parameters
    parser.add_argument("--chunk_sizes", nargs="+", type=int, default=[64, 128, 256],
                        help="Chunk sizes to test")
    parser.add_argument("--top_ks", nargs="+", type=int, default=[2, 4, 8],
                        help="Top-k values to test")
    parser.add_argument("--retrievers", nargs="+", type=str, default=["bm25", "dense", "hybrid"],
                        help="Retriever types to test")
    parser.add_argument("--configs_subset", action="store_true",
                        help="Run only 3 representative configs for quick testing")

    # Output settings
    parser.add_argument("--output_dir", type=str, default="extensions/2_retriever_surgery/results",
                        help="Output directory for results")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already completed configs")

    return parser.parse_args()


def generate_config_grid(chunk_sizes, top_ks, retrievers, subset=False) -> List[Tuple[int, int, str]]:
    """
    Generate grid of (chunk_size, top_k, retriever) configurations.

    If subset=True, return only 3 representative configs:
    - (128, 4, bm25) - baseline
    - (256, 8, dense) - dense + large chunks + many docs
    - (64, 2, hybrid) - hybrid + small chunks + few docs
    """
    if subset:
        return [
            (128, 4, "bm25"),
            (256, 8, "dense"),
            (64, 2, "hybrid"),
        ]

    # Full grid
    return list(itertools.product(chunk_sizes, top_ks, retrievers))


def create_prompts_subset(prompts_file: str, num_prompts: int, output_file: str):
    """Create a subset of prompts for testing."""
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)

    subset = prompts[:num_prompts]

    with open(output_file, 'w') as f:
        json.dump(subset, f, indent=2)

    print(f"Created prompts subset: {len(subset)} prompts -> {output_file}")
    return output_file


def run_config(chunk_size: int, top_k: int, retriever: str, args, prompts_file: str) -> Dict:
    """Run a single retriever configuration."""
    config_name = f"chunk{chunk_size}_k{top_k}_{retriever}"

    print(f"\n{'='*80}")
    print(f"Running config: {config_name}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Top-k: {top_k}")
    print(f"  Retriever: {retriever}")
    print(f"{'='*80}")

    # Prepare output paths
    output_subdir = os.path.join(args.output_dir, "outputs", config_name)
    os.makedirs(output_subdir, exist_ok=True)

    output_file = os.path.join(output_subdir, "results.json")

    # Check if already completed
    if args.resume and os.path.exists(output_file):
        print(f"✓ Config already completed, skipping (use --no-resume to rerun)")
        with open(output_file, 'r') as f:
            return json.load(f)

    # Build command
    cmd = [
        "python", "main.py",
        "--task", "io",
        "--api", args.api,
        "--hf_ckpt", args.hf_ckpt,
        "--is_chat_model", args.is_chat_model,
        "--io_input_path", prompts_file,
        "--io_output_root", output_subdir,
        "--raw_data_dir", args.raw_data_dir,
        "--datastore_root", "datastore",
        "--output_dir", output_subdir,  # Required by TrainingArguments
        "--max_retrieval_seq_length", str(chunk_size),
        "--k_for_ric", str(top_k),
        "--index_name", retriever,
    ]

    # Add Together API args if needed
    if args.api == "together":
        if args.together_ckpt is None:
            raise ValueError("--together_ckpt required when api=together")
        cmd.extend(["--together_ckpt", args.together_ckpt])

    # Add dense retriever args if needed
    if retriever in ["dense", "hybrid"]:
        cmd.extend([
            "--dense_model", "sentence-transformers/all-mpnet-base-v2",
            "--faiss_index_type", "Flat",
        ])

    if retriever == "hybrid":
        cmd.extend(["--hybrid_alpha", "0.5"])

    print(f"\nCommand: {' '.join(cmd)}\n")

    # Run command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        # Aggregate individual JSON files into results.json
        # main.py creates files at: output_subdir/{model_name}/{id}.json
        model_output_dir = os.path.join(output_subdir, args.hf_ckpt.split('/')[-1])
        
        if os.path.exists(model_output_dir):
            json_files = sorted([f for f in os.listdir(model_output_dir) if f.endswith('.json')])
            
            if json_files:
                # Load all individual results
                all_results = []
                for json_file in json_files:
                    json_path = os.path.join(model_output_dir, json_file)
                    with open(json_path, 'r') as f:
                        all_results.append(json.load(f))
                
                # Save consolidated results
                with open(output_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                
                print(f"✓ Config completed successfully ({len(all_results)} outputs)")
                return all_results
            else:
                print(f"✗ Warning: No JSON output files found in {model_output_dir}")
                return None
        else:
            print(f"✗ Warning: Model output directory not found at {model_output_dir}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"✗ Error running config {config_name}:")
        print(e.stderr)
        return None


def summarize_results(all_results: Dict[str, Dict], args, configs):
    """Create summary of all results."""
    summary_file = os.path.join(args.output_dir, "summary.json")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.hf_ckpt,
        "api": args.api,
        "num_prompts": args.num_prompts,
        "total_configs": len(configs),
        "configs": {}
    }

    for (chunk_size, top_k, retriever), results in zip(configs, all_results.values()):
        config_name = f"chunk{chunk_size}_k{top_k}_{retriever}"

        if results is None:
            continue

        summary["configs"][config_name] = {
            "chunk_size": chunk_size,
            "top_k": top_k,
            "retriever": retriever,
            "num_outputs": len(results) if isinstance(results, list) else 1,
            "output_path": os.path.join(args.output_dir, "outputs", config_name, "results.json")
        }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}")

    return summary


def main():
    args = parse_args()

    print(f"\n{'='*80}")
    print(f"Extension 2: Retriever Surgery - Retriever Sweep")
    print(f"{'='*80}")
    print(f"Model: {args.hf_ckpt}")
    print(f"API: {args.api}")
    print(f"Prompts: {args.prompts_file} (first {args.num_prompts})")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "outputs"), exist_ok=True)

    # Create prompts subset if needed
    if args.num_prompts < 100:
        prompts_subset_file = os.path.join(args.output_dir, f"prompts_{args.num_prompts}.json")
        if not os.path.exists(prompts_subset_file):
            create_prompts_subset(args.prompts_file, args.num_prompts, prompts_subset_file)
        prompts_file = prompts_subset_file
    else:
        prompts_file = args.prompts_file

    # Generate config grid
    configs = generate_config_grid(
        args.chunk_sizes,
        args.top_ks,
        args.retrievers,
        subset=args.configs_subset
    )

    print(f"Running {len(configs)} retriever configurations:\n")
    for i, (chunk_size, top_k, retriever) in enumerate(configs, 1):
        print(f"  {i}. chunk_size={chunk_size}, top_k={top_k}, retriever={retriever}")
    print()

    # Run each config
    all_results = {}
    for chunk_size, top_k, retriever in configs:
        config_name = f"chunk{chunk_size}_k{top_k}_{retriever}"
        results = run_config(chunk_size, top_k, retriever, args, prompts_file)
        all_results[config_name] = results

    # Summarize results
    summary = summarize_results(all_results, args, configs)

    print(f"\n{'='*80}")
    print(f"Retriever sweep completed!")
    print(f"Total configs: {len(configs)}")
    print(f"Successful: {sum(1 for r in all_results.values() if r is not None)}")
    print(f"Failed: {sum(1 for r in all_results.values() if r is None)}")
    print(f"\nNext step: Run analyze_positions.py to compute metrics and generate plots")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
