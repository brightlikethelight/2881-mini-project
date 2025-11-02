#!/usr/bin/env python3
"""
Extension 1: Prompt-Level Defenses - Defense Sweep Script

This script runs multiple defense configurations to measure utility vs leakage tradeoffs.
Each configuration applies different decoding constraints to prevent verbatim copying.

Usage:
    # Quick test with gpt2 (local)
    python extensions/1_prompt_defenses/scripts/run_defense_sweep.py \
        --api hf \
        --hf_ckpt gpt2 \
        --num_prompts 10 \
        --configs baseline ngram_3 bad_words

    # Full run with Llama-2-7B (Together API)
    python extensions/1_prompt_defenses/scripts/run_defense_sweep.py \
        --api together \
        --hf_ckpt meta-llama/Llama-2-7b-chat-hf \
        --together_ckpt togethercomputer/llama-2-7b-chat \
        --num_prompts 50 \
        --configs all
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Defense configurations
DEFENSE_CONFIGS = {
    "baseline": {
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 0,
        "use_bad_words_defense": False,
        "description": "No defense (baseline)"
    },
    "ngram_2": {
        "no_repeat_ngram_size": 2,
        "encoder_no_repeat_ngram_size": 0,
        "use_bad_words_defense": False,
        "description": "Block 2-gram repetition in output"
    },
    "ngram_3": {
        "no_repeat_ngram_size": 3,
        "encoder_no_repeat_ngram_size": 0,
        "use_bad_words_defense": False,
        "description": "Block 3-gram repetition in output"
    },
    "ngram_4": {
        "no_repeat_ngram_size": 4,
        "encoder_no_repeat_ngram_size": 0,
        "use_bad_words_defense": False,
        "description": "Block 4-gram repetition in output"
    },
    "ngram_5": {
        "no_repeat_ngram_size": 5,
        "encoder_no_repeat_ngram_size": 0,
        "use_bad_words_defense": False,
        "description": "Block 5-gram repetition in output"
    },
    "enc_ngram_2": {
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 2,
        "use_bad_words_defense": False,
        "description": "Block 2-grams from input appearing in output"
    },
    "enc_ngram_3": {
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 3,
        "use_bad_words_defense": False,
        "description": "Block 3-grams from input appearing in output"
    },
    "enc_ngram_4": {
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 4,
        "use_bad_words_defense": False,
        "description": "Block 4-grams from input appearing in output"
    },
    "enc_ngram_5": {
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 5,
        "use_bad_words_defense": False,
        "description": "Block 5-grams from input appearing in output"
    },
    "bad_words": {
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 0,
        "use_bad_words_defense": True,
        "bad_words_ngram_size": 4,
        "description": "Blacklist 4-grams from retrieved docs"
    },
    "bad_words_5": {
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 0,
        "use_bad_words_defense": True,
        "bad_words_ngram_size": 5,
        "description": "Blacklist 5-grams from retrieved docs"
    },
    "combined_light": {
        "no_repeat_ngram_size": 3,
        "encoder_no_repeat_ngram_size": 3,
        "use_bad_words_defense": False,
        "description": "Combined: ngram_3 + enc_ngram_3"
    },
    "combined_medium": {
        "no_repeat_ngram_size": 4,
        "encoder_no_repeat_ngram_size": 4,
        "use_bad_words_defense": False,
        "description": "Combined: ngram_4 + enc_ngram_4"
    },
    "combined_strong": {
        "no_repeat_ngram_size": 5,
        "encoder_no_repeat_ngram_size": 5,
        "use_bad_words_defense": False,
        "description": "Combined: ngram_5 + enc_ngram_5"
    },
    "combined_max": {
        "no_repeat_ngram_size": 5,
        "encoder_no_repeat_ngram_size": 5,
        "use_bad_words_defense": True,
        "bad_words_ngram_size": 5,
        "description": "Combined: all defenses at maximum strength"
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run defense sweep for Extension 1")

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

    # Defense selection
    parser.add_argument("--configs", nargs="+", default=["all"],
                        help="Defense configs to run (e.g., 'baseline ngram_3') or 'all'")

    # Output settings
    parser.add_argument("--output_dir", type=str, default="extensions/1_prompt_defenses/results",
                        help="Output directory for results")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already completed configs")

    return parser.parse_args()


def create_prompts_subset(prompts_file: str, num_prompts: int, output_file: str):
    """Create a subset of prompts for testing."""
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)

    subset = prompts[:num_prompts]

    with open(output_file, 'w') as f:
        json.dump(subset, f, indent=2)

    print(f"Created prompts subset: {len(subset)} prompts -> {output_file}")
    return output_file


def run_defense_config(config_name: str, config: Dict, args, prompts_file: str) -> Dict:
    """Run a single defense configuration."""
    print(f"\n{'='*80}")
    print(f"Running config: {config_name}")
    print(f"Description: {config['description']}")
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
        "--max_retrieval_seq_length", "256",
        "--k_for_ric", "3",
        "--no_repeat_ngram_size", str(config["no_repeat_ngram_size"]),
        "--encoder_no_repeat_ngram_size", str(config["encoder_no_repeat_ngram_size"]),
    ]

    # Add Together API args if needed
    if args.api == "together":
        if args.together_ckpt is None:
            raise ValueError("--together_ckpt required when api=together")
        cmd.extend(["--together_ckpt", args.together_ckpt])

    # Add bad_words_defense if enabled
    if config.get("use_bad_words_defense", False):
        cmd.extend([
            "--use_bad_words_defense", "True",
            "--bad_words_ngram_size", str(config.get("bad_words_ngram_size", 4))
        ])

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


def summarize_results(all_results: Dict[str, Dict], args):
    """Create summary table of all results."""
    summary_file = os.path.join(args.output_dir, "summary.json")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.hf_ckpt,
        "api": args.api,
        "num_prompts": args.num_prompts,
        "configs": {}
    }

    for config_name, results in all_results.items():
        if results is None:
            continue

        config_desc = DEFENSE_CONFIGS[config_name]["description"]

        # Extract metrics (will be computed by analyze_defenses.py)
        summary["configs"][config_name] = {
            "description": config_desc,
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
    print(f"Extension 1: Prompt-Level Defenses - Defense Sweep")
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

    # Determine which configs to run
    if "all" in args.configs:
        configs_to_run = list(DEFENSE_CONFIGS.keys())
    else:
        configs_to_run = args.configs

    print(f"Running {len(configs_to_run)} configs: {', '.join(configs_to_run)}\n")

    # Run each config
    all_results = {}
    for config_name in configs_to_run:
        if config_name not in DEFENSE_CONFIGS:
            print(f"Warning: Unknown config '{config_name}', skipping")
            continue

        config = DEFENSE_CONFIGS[config_name]
        results = run_defense_config(config_name, config, args, prompts_file)
        all_results[config_name] = results

    # Summarize results
    summary = summarize_results(all_results, args)

    print(f"\n{'='*80}")
    print(f"Defense sweep completed!")
    print(f"Total configs: {len(configs_to_run)}")
    print(f"Successful: {sum(1 for r in all_results.values() if r is not None)}")
    print(f"Failed: {sum(1 for r in all_results.values() if r is None)}")
    print(f"\nNext step: Run analyze_defenses.py to compute metrics and generate plots")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
