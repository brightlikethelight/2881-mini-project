"""
Main script to run comprehensive summarization comparison
Compares TextRank, LexRank, SumBasic, LSA, MMR, and LLM at 30%, 50%, 70% compression
"""

import asyncio
import json
from pathlib import Path
from comprehensive_summarization_comparison import (
    load_wiki_data, generate_summaries_async, evaluate_methods
)
from visualizations import plot_comprehensive_comparison

def print_evaluation_table(evaluation):
    """Print formatted evaluation table"""
    print("\n" + "="*130)
    print("COMPREHENSIVE SUMMARIZATION METHOD COMPARISON")
    print("="*130)
    print(f"{'Method':<25} {'Type':<12} {'ROUGE-L':<10} {'Privacy':<10} {'Compression':<12} {'Overlap':<10}")
    print(f"{'':25} {'':12} {'(Utility)':<10} {'Score':<10} {'Ratio':<12} {'(Risk)':<10}")
    print("-"*130)
    
    # Group by method type
    method_types = {
        'no_compression': 'Baseline',
        'textrank': 'Extractive',
        'lexrank': 'Extractive',
        'sumbasic': 'Extractive',
        'lsa': 'Extractive',
        'mmr': 'Extractive',
        'llm': 'Abstractive'
    }
    
    for method, metrics in evaluation.items():
        method_type = 'Unknown'
        for key in method_types:
            if key in method:
                method_type = method_types[key]
                break
        
        print(f"{method:<25} {method_type:<12} {metrics['rouge_l']:<10.3f} "
              f"{metrics['privacy_score']:<10.3f} {metrics['compression_ratio']:<12.3f} "
              f"{metrics['token_overlap']:<10.3f}")
    
    print("="*130)
    
    # Print top performers
    print("\n" + "="*80)
    print("TOP PERFORMERS")
    print("="*80)
    
    methods_only = {k: v for k, v in evaluation.items() if k != 'no_compression'}
    
    # Best utility
    best_utility = max(methods_only.items(), key=lambda x: x[1]['rouge_l'])
    print(f"\n🏆 Best Utility (ROUGE-L): {best_utility[0]}")
    print(f"   Score: {best_utility[1]['rouge_l']:.3f}")
    
    # Best privacy
    best_privacy = max(methods_only.items(), key=lambda x: x[1]['privacy_score'])
    print(f"\n🔒 Best Privacy: {best_privacy[0]}")
    print(f"   Score: {best_privacy[1]['privacy_score']:.3f}")
    
    # Best compression
    best_compression = min(methods_only.items(), key=lambda x: x[1]['compression_ratio'])
    print(f"\n📦 Best Compression: {best_compression[0]}")
    print(f"   Ratio: {best_compression[1]['compression_ratio']:.3f}")
    
    # Best balance (utility + privacy - compression)
    best_balance = max(methods_only.items(), 
                      key=lambda x: x[1]['rouge_l'] + x[1]['privacy_score'] - x[1]['compression_ratio'])
    print(f"\n⚖️  Best Balance: {best_balance[0]}")
    print(f"   Utility: {best_balance[1]['rouge_l']:.3f}, "
          f"Privacy: {best_balance[1]['privacy_score']:.3f}, "
          f"Compression: {best_balance[1]['compression_ratio']:.3f}")
    
    print("\n" + "="*80)

async def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARIZATION METHODS COMPARISON")
    print("Methods: TextRank, LexRank, SumBasic, LSA/SVD, MMR, LLM")
    print("Compression Ratios: 30%, 50%, 70%")
    print("="*80)
    
    # Configuration
    compression_ratios = [0.3, 0.5, 0.7]
    max_samples = 5
    
    # Load data
    print("\n🔄 Loading wiki data...")
    wiki_file = Path('raw_data/private/wiki_newest/wiki_newest.txt')
    
    if not wiki_file.exists():
        print(f"❌ Error: {wiki_file} not found")
        return
    
    samples = load_wiki_data(wiki_file, max_samples=max_samples)
    print(f"✓ Loaded {len(samples)} samples")
    
    # Generate summaries
    print(f"\n🔄 Generating summaries with all methods at {len(compression_ratios)} compression ratios...")
    print("   This will take several minutes due to LLM API calls...")
    results = await generate_summaries_async(samples, compression_ratios)
    print(f"✓ Generated summaries for {len(results)} samples")
    
    # Evaluate methods
    print("\n🔄 Evaluating all methods...")
    evaluation = evaluate_methods(results)
    
    # Print results
    print_evaluation_table(evaluation)
    
    # Save results
    output_dir = Path('results/comprehensive_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'evaluation': evaluation,
            'num_samples': len(results),
            'compression_ratios': compression_ratios,
            'detailed_results': results
        }, f, indent=2)
    print(f"\n✓ Detailed results saved to: {output_file}")
    
    # Generate visualizations
    print("\n🔄 Generating comprehensive visualizations...")
    plot_comprehensive_comparison(evaluation, output_dir)
    
    print("\n✅ Analysis complete!")
    print(f"📊 View all plots in: {output_dir}")
    print("\nGenerated visualizations:")
    print("  1. heatmap_all_methods.png - Heatmap of all methods and metrics")
    print("  2. privacy_utility_scatter.png - Privacy-utility trade-off scatter plot")
    print("  3. compression_comparison.png - Actual compression achieved")
    print("  4. comparison_by_ratio.png - Methods grouped by compression ratio")
    print("  5. radar_charts.png - Multi-metric radar comparison")
    print("  6. pareto_frontier.png - Pareto optimal methods")

if __name__ == "__main__":
    asyncio.run(main())
