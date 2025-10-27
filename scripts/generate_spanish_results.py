#!/usr/bin/env python3
"""
Generate Spanish RAG attack results analysis and comparison tables.

Usage:
    python scripts/generate_spanish_results.py \
        --results_dir eval_data/spanish/eval_results \
        --format both

Analyzes cross-lingual leakage patterns across Spanish datastore with different prompt types.
For Cross-Lingual RAG Copy-Out Attack Experiment
Based on ICLR 2025: "Follow My Instruction and Spill the Beans"
https://arxiv.org/abs/2402.17840
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def load_spanish_results(results_dir: str) -> Dict[str, Dict]:
    """Load all JSON result files and parse experiment metadata."""
    results = {}
    
    for json_file in Path(results_dir).glob("*.json"):
        model_name = json_file.stem
        
        with open(json_file) as f:
            metrics = json.load(f)
        
        # Parse experiment metadata from filename
        # Format: ModelName_prompttype.json
        parts = model_name.split('_')
        if len(parts) >= 2:
            base_model = '_'.join(parts[:-1])
            prompt_type = parts[-1]
        else:
            base_model = model_name
            prompt_type = "unknown"
        
        results[model_name] = {
            'base_model': base_model,
            'prompt_type': prompt_type,
            'metrics': metrics
        }
    
    return results


def format_spanish_comparison_table(results: Dict[str, Dict]) -> str:
    """Format results as comparison table showing cross-lingual patterns."""
    
    # Group by model and prompt type
    model_data = {}
    for exp_name, data in results.items():
        model = data['base_model']
        prompt_type = data['prompt_type']
        
        if model not in model_data:
            model_data[model] = {}
        
        model_data[model][prompt_type] = data['metrics']
    
    # Create comparison table
    table = "| Model | Prompt Type | ROUGE-L | BLEU | F1 | BERTScore |\n"
    table += "|-------|-------------|---------|------|----|-----------|\n"
    
    prompt_types = ['spanish', 'english', 'codeswitch']
    
    for model in sorted(model_data.keys()):
        model_short = model.split('/')[-1] if '/' in model else model
        
        for prompt_type in prompt_types:
            if prompt_type in model_data[model]:
                metrics = model_data[model][prompt_type]
                rouge = metrics.get('rougeL_score', 0) * 100
                bleu = metrics.get('bleu_score', 0)
                f1 = metrics.get('token_set_f1', 0) * 100
                bert = metrics.get('bert_score', 0) * 100
                
                table += f"| {model_short} | {prompt_type} | {rouge:.2f} | {bleu:.2f} | {f1:.2f} | {bert:.2f} |\n"
            else:
                table += f"| {model_short} | {prompt_type} | - | - | - | - |\n"
    
    return table


def format_language_comparison_table(results: Dict[str, Dict]) -> str:
    """Format results showing language-specific leakage patterns."""
    
    # Group by prompt type
    prompt_data = {}
    for exp_name, data in results.items():
        prompt_type = data['prompt_type']
        
        if prompt_type not in prompt_data:
            prompt_data[prompt_type] = []
        
        prompt_data[prompt_type].append({
            'model': data['base_model'],
            'metrics': data['metrics']
        })
    
    table = "| Prompt Type | Avg ROUGE-L | Avg BLEU | Avg F1 | Avg BERTScore | Count |\n"
    table += "|-------------|-------------|----------|--------|---------------|-------|\n"
    
    for prompt_type in ['spanish', 'english', 'codeswitch']:
        if prompt_type in prompt_data:
            experiments = prompt_data[prompt_type]
            
            avg_rouge = sum(exp['metrics'].get('rougeL_score', 0) for exp in experiments) / len(experiments) * 100
            avg_bleu = sum(exp['metrics'].get('bleu_score', 0) for exp in experiments) / len(experiments)
            avg_f1 = sum(exp['metrics'].get('token_set_f1', 0) for exp in experiments) / len(experiments) * 100
            avg_bert = sum(exp['metrics'].get('bert_score', 0) for exp in experiments) / len(experiments) * 100
            
            table += f"| {prompt_type} | {avg_rouge:.2f} | {avg_bleu:.2f} | {avg_f1:.2f} | {avg_bert:.2f} | {len(experiments)} |\n"
        else:
            table += f"| {prompt_type} | - | - | - | - | 0 |\n"
    
    return table


def format_model_comparison_table(results: Dict[str, Dict]) -> str:
    """Format results comparing multilingual vs English-centric models."""
    
    # Group by model
    model_data = {}
    for exp_name, data in results.items():
        model = data['base_model']
        
        if model not in model_data:
            model_data[model] = []
        
        model_data[model].append({
            'prompt_type': data['prompt_type'],
            'metrics': data['metrics']
        })
    
    table = "| Model | Type | Avg ROUGE-L | Avg BLEU | Avg F1 | Avg BERTScore |\n"
    table += "|-------|------|-------------|----------|--------|---------------|\n"
    
    for model in sorted(model_data.keys()):
        experiments = model_data[model]
        model_short = model.split('/')[-1] if '/' in model else model
        
        # Determine model type
        if 'qwen' in model.lower():
            model_type = "Multilingual"
        elif 'llama' in model.lower():
            model_type = "Multilingual"
        elif 'mistral' in model.lower():
            model_type = "English-centric"
        else:
            model_type = "Unknown"
        
        avg_rouge = sum(exp['metrics'].get('rougeL_score', 0) for exp in experiments) / len(experiments) * 100
        avg_bleu = sum(exp['metrics'].get('bleu_score', 0) for exp in experiments) / len(experiments)
        avg_f1 = sum(exp['metrics'].get('token_set_f1', 0) for exp in experiments) / len(experiments) * 100
        avg_bert = sum(exp['metrics'].get('bert_score', 0) for exp in experiments) / len(experiments) * 100
        
        table += f"| {model_short} | {model_type} | {avg_rouge:.2f} | {avg_bleu:.2f} | {avg_f1:.2f} | {avg_bert:.2f} |\n"
    
    return table


def generate_analysis_summary(results: Dict[str, Dict]) -> str:
    """Generate analysis summary with key findings."""
    
    summary = []
    summary.append("## Spanish RAG Copy-Out Attack Analysis")
    summary.append("")
    
    # Count experiments
    total_experiments = len(results)
    prompt_types = set(data['prompt_type'] for data in results.values())
    models = set(data['base_model'] for data in results.values())
    
    summary.append(f"**Total Experiments**: {total_experiments}")
    summary.append(f"**Models Tested**: {len(models)}")
    summary.append(f"**Prompt Types**: {', '.join(sorted(prompt_types))}")
    summary.append("")
    
    # Calculate averages by prompt type
    prompt_averages = {}
    for prompt_type in prompt_types:
        experiments = [data for data in results.values() if data['prompt_type'] == prompt_type]
        if experiments:
            avg_rouge = sum(exp['metrics'].get('rougeL_score', 0) for exp in experiments) / len(experiments) * 100
            prompt_averages[prompt_type] = avg_rouge
    
    summary.append("### Key Findings")
    summary.append("")
    
    if prompt_averages:
        # Find highest leakage
        best_prompt = max(prompt_averages.items(), key=lambda x: x[1])
        summary.append(f"**Highest Leakage**: {best_prompt[0]} prompts ({best_prompt[1]:.1f}% ROUGE-L)")
        
        # Compare Spanish vs English
        if 'spanish' in prompt_averages and 'english' in prompt_averages:
            spanish_score = prompt_averages['spanish']
            english_score = prompt_averages['english']
            diff = spanish_score - english_score
            summary.append(f"**Spanish vs English**: Spanish {spanish_score:.1f}% vs English {english_score:.1f}% ({diff:+.1f}% difference)")
        
        # Check code-switch effectiveness
        if 'codeswitch' in prompt_averages:
            codeswitch_score = prompt_averages['codeswitch']
            summary.append(f"**Code-switch Effectiveness**: {codeswitch_score:.1f}% ROUGE-L")
    
    summary.append("")
    summary.append("### Success Criteria Validation")
    summary.append("")
    
    # Check success criteria
    if prompt_averages:
        avg_leakage = sum(prompt_averages.values()) / len(prompt_averages)
        
        if avg_leakage > 30:
            summary.append("✅ **Language-agnostic leakage confirmed**: Average leakage >30% across prompt types")
        else:
            summary.append("⚠️ **Low leakage detected**: Average leakage <30%, may indicate language-specific effects")
        
        if 'codeswitch' in prompt_averages and prompt_averages['codeswitch'] > 30:
            summary.append("✅ **Code-switch robustness confirmed**: Non-trivial leakage under mixed-language prompts")
        else:
            summary.append("⚠️ **Code-switch limited**: Low leakage suggests language-specific tokenization effects")
    
    return "\n".join(summary)


def main():
    parser = argparse.ArgumentParser(
        description='Generate Spanish RAG attack results analysis',
        epilog='For Cross-Lingual RAG Copy-Out Attack Experiment'
    )
    parser.add_argument(
        '--results_dir',
        default='eval_data/spanish/eval_results',
        help='Directory with JSON result files (default: eval_data/spanish/eval_results)'
    )
    parser.add_argument(
        '--format',
        choices=['markdown', 'latex', 'both'],
        default='both',
        help='Output format (default: both)'
    )
    parser.add_argument(
        '--output',
        default='spanish_results.md',
        help='Output file (default: spanish_results.md)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Load results
    if verbose:
        print(f"Loading results from: {args.results_dir}")
    
    results = load_spanish_results(args.results_dir)
    
    if not results:
        print(f"⚠️  No results found in {args.results_dir}")
        return
    
    if verbose:
        print(f"✓ Loaded {len(results)} experiments")
        print(f"  Models: {len(set(data['base_model'] for data in results.values()))}")
        print(f"  Prompt types: {', '.join(set(data['prompt_type'] for data in results.values()))}")
        print()

    # Generate tables
    comparison_table = format_spanish_comparison_table(results)
    language_table = format_language_comparison_table(results)
    model_table = format_model_comparison_table(results)
    analysis_summary = generate_analysis_summary(results)

    # Combine all content
    full_content = []
    full_content.append(analysis_summary)
    full_content.append("")
    full_content.append("## Detailed Results")
    full_content.append("")
    full_content.append("### Cross-Lingual Comparison")
    full_content.append("")
    full_content.append(comparison_table)
    full_content.append("")
    full_content.append("### Language-Specific Patterns")
    full_content.append("")
    full_content.append(language_table)
    full_content.append("")
    full_content.append("### Model Comparison")
    full_content.append("")
    full_content.append(model_table)

    content = "\n".join(full_content)

    # Output
    if args.format in ['markdown', 'both']:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(content)
        
        if verbose:
            print(f"✓ Saved analysis to: {args.output}")

    if args.format in ['latex', 'both']:
        latex_output = args.output.replace('.md', '.tex')
        # Convert markdown tables to LaTeX (simplified)
        latex_content = content.replace('|', ' & ').replace('\n', ' \\\\\n')
        with open(latex_output, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        if verbose:
            print(f"✓ Saved LaTeX to: {latex_output}")

    if verbose:
        print()
        print("="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Key findings:")
        print("- Check if Spanish prompts show similar leakage to English")
        print("- Verify code-switch prompts maintain effectiveness")
        print("- Compare multilingual vs English-centric model performance")
        print("="*60)


if __name__ == '__main__':
    main()


