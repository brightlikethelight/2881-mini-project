#!/usr/bin/env python3
"""
Generate results tables from evaluation metrics.

Usage:
    python scripts/generate_results_table.py \
        --results_dir eval_data/wikipedia/eval_results \
        --format both

Formats results as Markdown and LaTeX tables for paper.
For ICLR 2025: "Follow My Instruction and Spill the Beans"
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_results(results_dir: str) -> Dict[str, Dict]:
    """Load all JSON result files from directory."""
    results = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    json_files = list(results_path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in: {results_dir}")

    for json_file in json_files:
        model_name = json_file.stem
        with open(json_file, 'r') as f:
            results[model_name] = json.load(f)

    return results


def shorten_model_name(model_name: str) -> str:
    """Shorten model name for display."""
    # Extract just the model name from paths like "meta-llama/Llama-2-7b-chat-hf"
    short_name = model_name.split('/')[-1]

    # Further abbreviations for common patterns
    replacements = {
        'Instruct': 'Inst',
        'instruct': 'inst',
        '-chat': '',
        '-hf': '',
    }

    for old, new in replacements.items():
        short_name = short_name.replace(old, new)

    return short_name


def format_markdown_table(results: Dict[str, Dict], sort_by: str = 'rougeL_score') -> str:
    """Format results as Markdown table."""
    # Sort models by specified metric (descending)
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1].get(sort_by, 0),
        reverse=True
    )

    # Build table
    table = "| Model | ROUGE-L | BLEU | F1 | BERTScore |\n"
    table += "|-------|---------|------|----|-----------|\\n"

    for model_name, metrics in sorted_models:
        # Extract metrics (multiply by 100 for percentages where appropriate)
        rouge = metrics.get('rougeL_score', 0) * 100
        bleu = metrics.get('bleu_score', 0)  # Already 0-100 scale
        f1 = metrics.get('token_set_f1', 0) * 100
        bert = metrics.get('bert_score', 0) * 100

        # Shorten model name
        display_name = shorten_model_name(model_name)

        table += f"| {display_name} | {rouge:.2f} | {bleu:.2f} | {f1:.2f} | {bert:.2f} |\n"

    return table


def format_markdown_table_with_sem(results: Dict[str, Dict], sort_by: str = 'rougeL_score') -> str:
    """Format results as Markdown table with standard errors."""
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1].get(sort_by, 0),
        reverse=True
    )

    table = "| Model | ROUGE-L | BLEU | F1 | BERTScore |\n"
    table += "|-------|---------|------|----|-----------|\\n"

    for model_name, metrics in sorted_models:
        # Extract metrics and SEMs
        rouge = metrics.get('rougeL_score', 0) * 100
        rouge_sem = metrics.get('rougeL_score_sem', 0) * 100

        bleu = metrics.get('bleu_score', 0)
        bleu_sem = metrics.get('bleu_score_sem', 0)

        f1 = metrics.get('token_set_f1', 0) * 100
        f1_sem = metrics.get('token_set_f1_sem', 0) * 100

        bert = metrics.get('bert_score', 0) * 100
        bert_sem = metrics.get('bert_score_sem', 0) * 100

        display_name = shorten_model_name(model_name)

        table += f"| {display_name} | {rouge:.2f}±{rouge_sem:.2f} | {bleu:.2f}±{bleu_sem:.2f} | {f1:.2f}±{f1_sem:.2f} | {bert:.2f}±{bert_sem:.2f} |\n"

    return table


def format_latex_table(results: Dict[str, Dict], sort_by: str = 'rougeL_score') -> str:
    """Format results as LaTeX table."""
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1].get(sort_by, 0),
        reverse=True
    )

    table = "\\begin{table}[h]\n"
    table += "\\centering\n"
    table += "\\begin{tabular}{lcccc}\n"
    table += "\\toprule\n"
    table += "Model & ROUGE-L & BLEU & F1 & BERTScore \\\\\\\n"
    table += "\\midrule\n"

    for model_name, metrics in sorted_models:
        rouge = metrics.get('rougeL_score', 0) * 100
        bleu = metrics.get('bleu_score', 0)
        f1 = metrics.get('token_set_f1', 0) * 100
        bert = metrics.get('bert_score', 0) * 100

        # Escape underscores for LaTeX
        display_name = shorten_model_name(model_name).replace('_', '\\_')

        # Bold the best score in each column
        table += f"{display_name} & {rouge:.2f} & {bleu:.2f} & {f1:.2f} & {bert:.2f} \\\\\\\n"

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\\caption{RAG Copy-Out Attack Results. Higher scores indicate more successful data extraction.}\n"
    table += "\\label{tab:rag_attack_results}\n"
    table += "\\end{table}\n"

    return table


def format_csv(results: Dict[str, Dict], sort_by: str = 'rougeL_score') -> str:
    """Format results as CSV."""
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1].get(sort_by, 0),
        reverse=True
    )

    csv = "Model,ROUGE-L,BLEU,F1,BERTScore\n"

    for model_name, metrics in sorted_models:
        rouge = metrics.get('rougeL_score', 0) * 100
        bleu = metrics.get('bleu_score', 0)
        f1 = metrics.get('token_set_f1', 0) * 100
        bert = metrics.get('bert_score', 0) * 100

        csv += f"{model_name},{rouge:.2f},{bleu:.2f},{f1:.2f},{bert:.2f}\n"

    return csv


def print_summary_stats(results: Dict[str, Dict]):
    """Print summary statistics across all models."""
    if not results:
        return

    # Collect all metric values
    rouge_scores = [m.get('rougeL_score', 0) * 100 for m in results.values()]
    bleu_scores = [m.get('bleu_score', 0) for m in results.values()]
    f1_scores = [m.get('token_set_f1', 0) * 100 for m in results.values()]
    bert_scores = [m.get('bert_score', 0) * 100 for m in results.values()]

    print("\nSummary Statistics (across all models):")
    print(f"  ROUGE-L:   min={min(rouge_scores):.2f}, max={max(rouge_scores):.2f}, avg={sum(rouge_scores)/len(rouge_scores):.2f}")
    print(f"  BLEU:      min={min(bleu_scores):.2f}, max={max(bleu_scores):.2f}, avg={sum(bleu_scores)/len(bleu_scores):.2f}")
    print(f"  F1:        min={min(f1_scores):.2f}, max={max(f1_scores):.2f}, avg={sum(f1_scores)/len(f1_scores):.2f}")
    print(f"  BERTScore: min={min(bert_scores):.2f}, max={max(bert_scores):.2f}, avg={sum(bert_scores)/len(bert_scores):.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate results tables from evaluation metrics',
        epilog='For ICLR 2025 paper: "Follow My Instruction and Spill the Beans"'
    )
    parser.add_argument(
        '--results_dir',
        default='eval_data/wikipedia/eval_results',
        help='Directory with JSON result files (default: eval_data/wikipedia/eval_results)'
    )
    parser.add_argument(
        '--format',
        choices=['markdown', 'latex', 'csv', 'both', 'all'],
        default='both',
        help='Output format (default: both)'
    )
    parser.add_argument(
        '--sort_by',
        choices=['rougeL_score', 'bleu_score', 'token_set_f1', 'bert_score'],
        default='rougeL_score',
        help='Sort models by this metric (default: rougeL_score)'
    )
    parser.add_argument(
        '--output_prefix',
        default='results_table',
        help='Output file prefix (default: results_table)'
    )
    parser.add_argument(
        '--show_sem',
        action='store_true',
        help='Show standard errors in Markdown table'
    )

    args = parser.parse_args()

    try:
        # Load results
        results = load_results(args.results_dir)
        print(f"✓ Loaded results for {len(results)} models from: {args.results_dir}")

        # Print summary stats
        print_summary_stats(results)
        print()

        # Generate tables
        if args.format in ['markdown', 'both', 'all']:
            print("=" * 60)
            print("MARKDOWN TABLE")
            print("=" * 60)
            if args.show_sem:
                md_table = format_markdown_table_with_sem(results, args.sort_by)
            else:
                md_table = format_markdown_table(results, args.sort_by)
            print(md_table)
            print()

            # Save to file
            md_file = f"{args.output_prefix}.md"
            with open(md_file, 'w') as f:
                f.write(md_table)
            print(f"✓ Saved Markdown table to: {md_file}\n")

        if args.format in ['latex', 'both', 'all']:
            print("=" * 60)
            print("LATEX TABLE")
            print("=" * 60)
            latex_table = format_latex_table(results, args.sort_by)
            print(latex_table)
            print()

            # Save to file
            latex_file = f"{args.output_prefix}.tex"
            with open(latex_file, 'w') as f:
                f.write(latex_table)
            print(f"✓ Saved LaTeX table to: {latex_file}\n")

        if args.format in ['csv', 'all']:
            csv_table = format_csv(results, args.sort_by)
            csv_file = f"{args.output_prefix}.csv"
            with open(csv_file, 'w') as f:
                f.write(csv_table)
            print(f"✓ Saved CSV table to: {csv_file}\n")

        print("=" * 60)
        print("Next steps:")
        print("  1. Copy tables to your paper")
        print("  2. Interpret results (higher scores = more data leakage)")
        print("  3. Compare with paper Table 1")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
