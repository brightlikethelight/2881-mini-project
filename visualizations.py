"""
Visualization functions for comprehensive summarization comparison
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def plot_comprehensive_comparison(evaluation, output_dir):
    """Create comprehensive visualizations"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate methods by type and ratio
    methods = list(evaluation.keys())
    
    # Plot 1: Heatmap of all metrics
    plot_heatmap(evaluation, output_dir)
    
    # Plot 2: Privacy-Utility scatter by method type
    plot_privacy_utility_scatter(evaluation, output_dir)
    
    # Plot 3: Compression ratio comparison
    plot_compression_comparison(evaluation, output_dir)
    
    # Plot 4: Method comparison by compression ratio
    plot_by_compression_ratio(evaluation, output_dir)
    
    # Plot 5: Radar charts for each compression ratio
    plot_radar_charts(evaluation, output_dir)
    
    # Plot 6: Pareto frontier
    plot_pareto_frontier(evaluation, output_dir)

def plot_heatmap(evaluation, output_dir):
    """Heatmap showing all methods and metrics"""
    methods = [m for m in evaluation.keys() if m != 'no_compression']
    metrics = ['rouge_l', 'privacy_score', 'compression_ratio', 'token_overlap']
    metric_labels = ['ROUGE-L\n(Utility)\n↑ Better', 'Privacy\nScore\n↑ Better', 
                     'Compression\nRatio\n↓ Better', 'Token\nOverlap\n↓ Better']
    
    # Create data matrix
    data = np.array([[evaluation[method][metric] for metric in metrics] for method in methods])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_yticklabels(methods, fontsize=9)
    
    # Add values to cells
    for i in range(len(methods)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8, fontweight='bold')
    
    ax.set_title('Comprehensive Method Comparison Heatmap', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.colorbar(im, ax=ax, label='Score (0-1)')
    plt.tight_layout()
    
    plot_path = output_dir / 'heatmap_all_methods.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plot_path}")

def plot_privacy_utility_scatter(evaluation, output_dir):
    """Privacy-Utility scatter plot with method types"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define colors and markers by method type
    method_styles = {
        'textrank': {'color': '#3498db', 'marker': 'o', 'label': 'TextRank'},
        'lexrank': {'color': '#2ecc71', 'marker': 's', 'label': 'LexRank'},
        'sumbasic': {'color': '#f39c12', 'marker': '^', 'label': 'SumBasic'},
        'lsa': {'color': '#9b59b6', 'marker': 'D', 'label': 'LSA'},
        'mmr': {'color': '#1abc9c', 'marker': 'v', 'label': 'MMR'},
        'llm': {'color': '#e74c3c', 'marker': '*', 'label': 'LLM'},
    }
    
    plotted_labels = set()
    
    for method_name, metrics in evaluation.items():
        if method_name == 'no_compression':
            continue
        
        # Determine method type
        method_type = method_name.split('_')[0]
        style = method_styles.get(method_type, {'color': 'gray', 'marker': 'o', 'label': method_type})
        
        # Determine size by compression ratio
        if '30%' in method_name:
            size = 200
        elif '50%' in method_name:
            size = 300
        elif '70%' in method_name:
            size = 400
        else:
            size = 150
        
        privacy = metrics['privacy_score']
        utility = metrics['rouge_l']
        
        label = style['label'] if style['label'] not in plotted_labels else None
        if label:
            plotted_labels.add(label)
        
        ax.scatter(privacy, utility, s=size, marker=style['marker'], 
                  color=style['color'], alpha=0.7, edgecolors='black', 
                  linewidth=1.5, label=label)
        
        # Add method name annotation
        ax.annotate(method_name.replace('_', '\n'), (privacy, utility), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=7, alpha=0.7)
    
    ax.set_xlabel('Privacy Score (Higher = Better Privacy)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Utility (ROUGE-L F1)', fontsize=13, fontweight='bold')
    ax.set_title('Privacy-Utility Trade-off Across All Methods\n(Size indicates compression ratio: larger = more compression)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plot_path = output_dir / 'privacy_utility_scatter.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plot_path}")

def plot_compression_comparison(evaluation, output_dir):
    """Compare actual compression achieved by each method"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Define method type colors
    method_colors = {
        'textrank': '#3498db',
        'lexrank': '#2ecc71',
        'sumbasic': '#f39c12',
        'lsa': '#9b59b6',
        'mmr': '#e74c3c',
        'llm': '#34495e'
    }
    
    # Get all methods and sort by compression ratio
    methods = [m for m in evaluation.keys() if m != 'no_compression']
    methods = sorted(methods, key=lambda x: evaluation[x]['compression_ratio'])
    
    compression_ratios = [evaluation[m]['compression_ratio'] for m in methods]
    
    # Assign colors by method type
    colors = []
    for method in methods:
        method_type = method.split('_')[0]
        colors.append(method_colors.get(method_type, 'gray'))
    
    bars = ax.bar(range(len(methods)), compression_ratios, color=colors, 
                  alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Compression Ratio (Lower = More Compression)', fontsize=12, fontweight='bold')
    ax.set_title('Actual Compression Achieved by Each Method', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=method_type.upper(), alpha=0.7)
                      for method_type, color in method_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, title='Method Type', 
             title_fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plot_path = output_dir / 'compression_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plot_path}")

def plot_by_compression_ratio(evaluation, output_dir):
    """Compare methods grouped by compression ratio with methods as sub-groups"""
    # Extract all method types and compression ratios
    methods = set()
    ratios = set()
    data = {}
    
    for method_name, metrics in evaluation.items():
        if method_name == 'no_compression':
            continue
            
        # Split method name into type and ratio
        parts = method_name.split('_')
        if len(parts) >= 2 and parts[-1].endswith('%'):
            method = '_'.join(parts[:-1])
            ratio = parts[-1]
            methods.add(method)
            ratios.add(ratio)
            if ratio not in data:
                data[ratio] = {}
            data[ratio][method] = metrics
    
    if not data:
        return
    
    # Sort ratios and methods for consistent ordering
    ratios = sorted(ratios, key=lambda x: int(x.rstrip('%')))
    methods = sorted(methods)
    
    # Define distinct colors for each method
    method_colors = {
        'textrank': '#3498db',    # Blue
        'lexrank': '#2ecc71',     # Green
        'sumbasic': '#f39c12',   # Orange
        'lsa': '#9b59b6',        # Purple
        'mmr': '#e74c3c',        # Red (changed from teal to be more distinct)
        'llm': '#1abc9c',        # Teal (moved from MMR)
    }
    
    # Set up the plot
    n_ratios = len(ratios)
    width = 0.15  # Width of each bar
    spacing = 0.02  # Spacing between bars
    group_width = len(methods) * (width + spacing) - spacing
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Method Comparison by Compression Ratio', fontsize=15, fontweight='bold')
    
    # Calculate x positions for each ratio group
    x = np.arange(n_ratios)
    
    # Plot ROUGE-L (Utility) scores
    for i, method in enumerate(methods):
        scores = [data[ratio].get(method, {}).get('rouge_l', 0) for ratio in ratios]
        pos = x - group_width/2 + i * (width + spacing) + width/2
        ax1.bar(pos, scores, width, 
               label=method.capitalize(),
               color=method_colors.get(method, 'gray'),
               alpha=0.8,
               edgecolor='black',
               linewidth=0.7)
        
        # Add value labels
        for j, score in enumerate(scores):
            if score > 0:  # Only add label if score exists
                ax1.text(pos[j], score + 0.02, 
                       f'{score:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Plot Privacy scores
    for i, method in enumerate(methods):
        scores = [data[ratio].get(method, {}).get('privacy_score', 0) for ratio in ratios]
        pos = x - group_width/2 + i * (width + spacing) + width/2
        ax2.bar(pos, scores, width, 
               label=method.capitalize(),
               color=method_colors.get(method, 'gray'),
               alpha=0.8,
               edgecolor='black',
               linewidth=0.7)
        
        # Add value labels
        for j, score in enumerate(scores):
            if score > 0:  # Only add label if score exists
                ax2.text(pos[j], score + 0.02, 
                       f'{score:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Customize the plots
    for ax, title in zip([ax1, ax2], ['ROUGE-L (Utility)', 'Privacy Score']):
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.2, axis='y', linestyle='--')
        ax.legend(title='Method', fontsize=10, title_fontsize=11, 
                 bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set x-axis labels and ticks
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{ratio} Compression' for ratio in ratios], 
                       fontsize=11, fontweight='bold')
    ax2.set_xlabel('Compression Level', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.97])  # Adjust right margin for legend
    plot_path = output_dir / 'comparison_by_ratio.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plot_path}")

def plot_radar_charts(evaluation, output_dir):
    """Radar charts comparing methods at each compression ratio"""
    ratios = ['30%', '50%', '70%']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))
    fig.suptitle('Multi-Metric Radar Comparison by Compression Ratio', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    metrics = ['rouge_l', 'privacy_score']
    metric_labels = ['Utility\n(ROUGE-L)', 'Privacy\nScore']
    
    for idx, ratio in enumerate(ratios):
        ax = axes[idx]
        
        methods_for_ratio = [m for m in evaluation.keys() if ratio in m and m != 'no_compression']
        
        if not methods_for_ratio:
            continue
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e74c3c']
        
        for i, method in enumerate(methods_for_ratio):
            values = [evaluation[method][m] for m in metrics]
            values += values[:1]
            
            color = colors[i % len(colors)]
            method_label = method.split('_')[0]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method_label, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(f'{ratio} Compression', fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.grid(True)
    
    plt.tight_layout()
    plot_path = output_dir / 'radar_charts.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plot_path}")

def plot_pareto_frontier(evaluation, output_dir):
    """Plot Pareto frontier showing optimal privacy-utility trade-offs"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    methods = [m for m in evaluation.keys() if m != 'no_compression']
    
    privacy_scores = []
    utility_scores = []
    method_names = []
    
    for method in methods:
        privacy_scores.append(evaluation[method]['privacy_score'])
        utility_scores.append(evaluation[method]['rouge_l'])
        method_names.append(method)
    
    # Find Pareto frontier
    pareto_indices = []
    for i in range(len(methods)):
        is_pareto = True
        for j in range(len(methods)):
            if i != j:
                if (privacy_scores[j] >= privacy_scores[i] and utility_scores[j] >= utility_scores[i] and
                    (privacy_scores[j] > privacy_scores[i] or utility_scores[j] > utility_scores[i])):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_indices.append(i)
    
    # Plot all points
    colors = []
    for method in methods:
        if 'llm' in method:
            colors.append('#e74c3c')
        elif 'textrank' in method:
            colors.append('#3498db')
        elif 'lexrank' in method:
            colors.append('#2ecc71')
        elif 'sumbasic' in method:
            colors.append('#f39c12')
        elif 'lsa' in method:
            colors.append('#9b59b6')
        elif 'mmr' in method:
            colors.append('#1abc9c')
        else:
            colors.append('gray')
    
    for i, method in enumerate(methods):
        ax.scatter(privacy_scores[i], utility_scores[i], s=200, 
                  color=colors[i], alpha=0.6, edgecolors='black', linewidth=1.5)
        ax.annotate(method, (privacy_scores[i], utility_scores[i]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)
    
    # Highlight Pareto frontier
    pareto_privacy = [privacy_scores[i] for i in pareto_indices]
    pareto_utility = [utility_scores[i] for i in pareto_indices]
    
    # Sort for line plotting
    sorted_pairs = sorted(zip(pareto_privacy, pareto_utility))
    pareto_privacy_sorted = [p[0] for p in sorted_pairs]
    pareto_utility_sorted = [p[1] for p in sorted_pairs]
    
    ax.plot(pareto_privacy_sorted, pareto_utility_sorted, 'r--', linewidth=3, 
           label='Pareto Frontier', alpha=0.7)
    
    for i in pareto_indices:
        ax.scatter(privacy_scores[i], utility_scores[i], s=400, 
                  facecolors='none', edgecolors='red', linewidth=3)
    
    ax.set_xlabel('Privacy Score (Higher = Better)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Utility (ROUGE-L)', fontsize=13, fontweight='bold')
    ax.set_title('Pareto Frontier: Optimal Privacy-Utility Trade-offs\n(Red circles = Pareto optimal methods)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plot_path = output_dir / 'pareto_frontier.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plot_path}")
