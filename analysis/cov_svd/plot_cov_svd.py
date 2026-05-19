import torch
import numpy as np
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse

# Add root for path consistency if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

def plot_spectrum(n_shuffles_list, svs_cov_dict, svs_corr_dict, output_dir, title_prefix, percentile=None, show_cov=True, show_corr=True):
    num_plots = int(show_cov) + int(show_corr)
    if num_plots == 0: return
    
    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 7), squeeze=False)
    axes = axes.flatten()
    
    first_key = n_shuffles_list[0]
    num_svs = svs_cov_dict[first_key].shape[1] if show_cov else svs_corr_dict[first_key].shape[1]
    k_max = num_svs if percentile is None else int(num_svs * percentile)
    
    suffix = f"_top{int(percentile*100)}" if percentile is not None else ""
    title_suffix = f" (Top {int(percentile*100)}%)" if percentile is not None else " (Full)"
    
    plot_configs = []
    if show_cov:
        plot_configs.append((svs_cov_dict, "Covariance" + title_suffix))
    if show_corr:
        plot_configs.append((svs_corr_dict, "Correlation" + title_suffix))
    
    for ax, (data_dict, title) in zip(axes, plot_configs):
        palette = sns.color_palette("viridis", n_colors=len(n_shuffles_list))
        for i, n_shuffles in enumerate(n_shuffles_list):
            data = data_dict[n_shuffles][:, :k_max]
            x = np.arange(1, k_max + 1)
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            color = palette[i]
            
            ax.plot(x, mean, label=f"{n_shuffles}", color=color, linewidth=2, alpha=0.8)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.1)
            
        ax.set_xlabel("Singular Value Index $k$", fontweight='bold')
        ax.set_ylabel("Singular Value $\sigma_k$", fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_xlim(1, k_max)
        ax.set_ylim(0, None)
        leg = ax.legend(title="Shuffles", loc='upper right', ncol=2, fontsize='small')
        plt.setp(leg.get_title(), fontweight='bold')
    
    plt.suptitle(f"{title_prefix}: Singular Value Spectrum{title_suffix}", fontsize=18, fontweight='bold', y=1.02)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cov_svd_spectrum{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'cov_svd_spectrum{suffix}.pdf'), bbox_inches='tight')
    plt.close()

def plot_cumulative_spectrum(n_shuffles_list, svs_cov_dict, svs_corr_dict, output_dir, title_prefix, percentile=None, show_cov=True, show_corr=True):
    num_plots = int(show_cov) + int(show_corr)
    if num_plots == 0: return
    
    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 7), squeeze=False)
    axes = axes.flatten()
    
    first_key = n_shuffles_list[0]
    num_svs = svs_cov_dict[first_key].shape[1] if show_cov else svs_corr_dict[first_key].shape[1]
    k_max = num_svs if percentile is None else int(num_svs * percentile)
    
    suffix = f"_top{int(percentile*100)}" if percentile is not None else ""
    title_suffix = f" (Top {int(percentile*100)}%)" if percentile is not None else " (Full)"
    
    plot_configs = []
    if show_cov:
        plot_configs.append((svs_cov_dict, "Covariance Partial Sums" + title_suffix))
    if show_corr:
        plot_configs.append((svs_corr_dict, "Correlation Partial Sums" + title_suffix))
    
    for ax, (data_dict, title) in zip(axes, plot_configs):
        palette = sns.color_palette("viridis", n_colors=len(n_shuffles_list))
        for i, n_shuffles in enumerate(n_shuffles_list):
            data = data_dict[n_shuffles][:, :k_max]
            # Compute cumulative sum along singular values
            cum_data = np.cumsum(data, axis=1)
            
            x = np.arange(1, k_max + 1)
            mean = np.mean(cum_data, axis=0)
            std = np.std(cum_data, axis=0)
            color = palette[i]
            
            ax.plot(x, mean, label=f"{n_shuffles}", color=color, linewidth=2, alpha=0.8)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.1)
            
        ax.set_xlabel("Singular Value Index $k$", fontweight='bold')
        ax.set_ylabel("Partial Sum $\sum_{i=1}^k \sigma_i$", fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_xlim(1, k_max)
        ax.set_ylim(0, None)
        leg = ax.legend(title="Shuffles", loc='lower right', ncol=2, fontsize='small')
        plt.setp(leg.get_title(), fontweight='bold')
    
    plt.suptitle(f"{title_prefix}: Cumulative Singular Value Spectrum{title_suffix}", fontsize=18, fontweight='bold', y=1.02)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cov_svd_cumulative_spectrum{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'cov_svd_cumulative_spectrum{suffix}.pdf'), bbox_inches='tight')
    plt.close()

def plot_summary_metric(x_summary, y_means_cov, y_stds_cov, y_means_corr, y_stds_corr, title, ylabel, filename, col1, col2, output_dir, show_cov=True, show_corr=True):
    num_plots = int(show_cov) + int(show_corr)
    if num_plots == 0: return
    
    fig, axes = plt.subplots(1, num_plots, figsize=(7.5 * num_plots, 6), squeeze=False)
    axes = axes.flatten()
    
    data_list = []
    if show_cov:
        data_list.append((y_means_cov, y_stds_cov, col1, "Covariance"))
    if show_corr:
        data_list.append((y_means_corr, y_stds_corr, col2, "Correlation"))
    
    for ax, (means, stds, color, t) in zip(axes, data_list):
        # Plot mean line with dots
        ax.plot(x_summary, means, '-o', color=color, linewidth=2.5, markersize=8, label=ylabel)
        # Plot confidence interval using fill_between
        ax.fill_between(x_summary, means - stds, means + stds, color=color, alpha=0.2)
        
        ax.set_xscale('log')
        ax.set_xlabel("Number of Shuffles (log scale)", fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(t, fontweight='bold')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        sns.despine(ax=ax)

    plt.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, filename + '.pdf'), bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(x_summary, y_means1, y_stds1, y_means2, y_stds2, title, ylabel1, ylabel2, label1, label2, color1, color2, filename, output_dir):
    fig, ax1 = plt.subplots(figsize=(11, 7.5))
    
    # Plot Metric 1 (Left Y)
    ax1.set_xscale('log')
    ax1.set_xlabel("Number of Shuffles (log scale)", fontweight='bold', fontsize=14)
    ax1.set_ylabel(ylabel1, color=color1, fontweight='bold', fontsize=14)
    lns1 = ax1.plot(x_summary, y_means1, '-o', color=color1, linewidth=3, markersize=12, label=label1, alpha=0.85)
    ax1.fill_between(x_summary, y_means1 - y_stds1, y_means1 + y_stds1, color=color1, alpha=0.15)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
    
    # Create Metric 2 (Right Y) - Double Y
    ax2 = ax1.twinx()
    ax2.set_ylabel(ylabel2, color=color2, fontweight='bold', fontsize=14)
    lns2 = ax2.plot(x_summary, y_means2, '-s', color=color2, linewidth=3, markersize=12, label=label2, alpha=0.85)
    ax2.fill_between(x_summary, y_means2 - y_stds2, y_means2 + y_stds2, color=color2, alpha=0.15)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)
    
    # Align first points (n_shuffle=0) proportionally
    v1_0, v2_0 = y_means1[0], y_means2[0]
    y1_max = v1_0 * 1.15 if v1_0 > 0 else 1.0
    y2_max = v2_0 * 1.15 if v2_0 > 0 else 1.0
    
    ax1.set_ylim(0, y1_max)
    ax2.set_ylim(0, y2_max)
    
    # Refined Grid
    ax1.grid(True, which='both', linestyle='--', alpha=0.4)
    
    # Combined high-quality legend
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    leg = ax1.legend(lns, labs, loc='upper right', frameon=True, fontsize=12, shadow=True)
    plt.setp(leg.get_frame(), alpha=1.0)
    
    plt.title(title, fontsize=20, fontweight='bold', pad=25)
    sns.despine(ax=ax1, right=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, filename + '.pdf'), bbox_inches='tight')
    plt.close()

def plot_polished_norms_comparison(x_summary, y_means_nn, y_stds_nn, y_means_fro2, y_stds_fro2, title, output_dir):
    fig, ax1 = plt.subplots(figsize=(10, 8))
    
    color_nn = '#801818' # Deep maroon
    color_fro2 = '#ff911a' # Vibrant orange
    
    # Plot Nuclear Norm (Left Y)
    ax1.set_xscale('log')
    ax1.set_xlabel("Number of Shuffles (log scale)", fontweight='bold', fontsize=14, color='#333333')
    ax1.set_ylabel(r"Nuclear Norm $\sum \sigma_k$", color=color_nn, fontweight='bold', fontsize=16)
    
    lns1 = ax1.plot(x_summary, y_means_nn, '-o', color=color_nn, linewidth=3.5, markersize=14, label="Nuclear Norm", alpha=0.9)
    ax1.fill_between(x_summary, y_means_nn - y_stds_nn, y_means_nn + y_stds_nn, color=color_nn, alpha=0.1)
    ax1.tick_params(axis='y', labelcolor=color_nn, labelsize=13)
    
    # Plot Squared Frobenius (Right Y)
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"Squared Frobenius $\sum \sigma_k^2$", color=color_fro2, fontweight='bold', fontsize=16)
    
    lns2 = ax2.plot(x_summary, y_means_fro2, '-s', color=color_fro2, linewidth=3.5, markersize=14, label="Squared Frobenius", alpha=0.9)
    ax2.fill_between(x_summary, y_means_fro2 - y_stds_fro2, y_means_fro2 + y_stds_fro2, color=color_fro2, alpha=0.1)
    ax2.tick_params(axis='y', labelcolor=color_fro2, labelsize=13)
    
    # Annotations
    idx0 = np.where(x_summary == 0.5)[0][0]
    
    # Box for MCP_O(d)
    ax1.annotate(r"MCP$_{O(d)}$", xy=(x_summary[idx0] * 1.2, y_means_nn[idx0] + 2), xytext=(20, y_means_nn[idx0]*1.05),
                 color=color_nn, fontweight='bold', fontsize=18,
                 bbox=dict(boxstyle="square,pad=0.3", fc="white", ec=color_nn, lw=2),
                 arrowprops=dict(arrowstyle="->", color=color_nn, lw=2.5, connectionstyle="arc3,rad=0.05", shrinkB=8))
    
    # Box for MCP_SF
    ax2.annotate(r"MCP$_{S_F}$", xy=(x_summary[idx0] * 1, y_means_fro2[idx0] - 1), xytext=(0.8, y_means_fro2[idx0]*0.65),
                 color=color_fro2, fontweight='bold', fontsize=18,
                 bbox=dict(boxstyle="square,pad=0.3", fc="white", ec=color_fro2, lw=1.5),
                 arrowprops=dict(arrowstyle="->", color=color_fro2, lw=2.5, connectionstyle="arc3,rad=-0.1", shrinkB=8))

    # Limits and Grid
    ax1.set_ylim(0, max(y_means_nn) * 1.15)
    ax2.set_ylim(0, max(y_means_fro2) * 1.15)
    
    ax1.grid(True, which='major', linestyle='--', alpha=0.4, color='#dddddd')
    ax1.grid(True, which='minor', linestyle=':', alpha=0.2, color='#eeeeee')
    
    # Legend
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    leg = ax1.legend(lns, labs, loc='upper right', frameon=True, fontsize=13, shadow=True, borderpad=1)
    plt.setp(leg.get_frame(), edgecolor='#cccccc')
    
    plt.title(title, fontsize=24, fontweight='bold', pad=35, color='#222222')
    sns.despine(ax=ax1, right=False)
    plt.tight_layout()
    
    filename = "cov_svd_norms_comparison_polished"
    plt.savefig(os.path.join(output_dir, filename + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, filename + '.pdf'), bbox_inches='tight')
    plt.close()

def plot_analysis(args):
    dataset_name = args.dataset
    view1_idx = args.view1_idx
    view2_idx = args.view2_idx
    
    show_cov = False
    show_corr = True
        
    save_name = dataset_name
    if view1_idx is not None and view2_idx is not None:
        save_name = f"{dataset_name}_{view1_idx}_{view2_idx}"
    
    title_prefix = dataset_name.upper()
    if view1_idx is not None and view2_idx is not None:
        title_prefix += f" ({view1_idx}, {view2_idx})"
        
    # Load results
    results_path = os.path.join(project_root, 'results/cov_svd', save_name, 'results.pt')
    if not os.path.exists(results_path):
        print(f"Results not found at {results_path}. Please run cov_svd.py first.")
        return
        
    results = torch.load(results_path, weights_only=False)
    n_shuffles_list = results['n_shuffles_list']
    svs_cov_dict = None
    svs_corr_dict = results['svs_corr']
    
    # Set style for academic figures
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": '--',
    })
    sns.set_style("whitegrid")
    
    output_dir = os.path.join(project_root, 'analysis/cov_svd/figures', save_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. SV Spectrum (Full)
    plot_spectrum(n_shuffles_list, svs_cov_dict, svs_corr_dict, output_dir, title_prefix, percentile=None, show_cov=show_cov, show_corr=show_corr)
    
    # 2. SV Spectrum (Truncated 10%)
    plot_spectrum(n_shuffles_list, svs_cov_dict, svs_corr_dict, output_dir, title_prefix, percentile=0.1, show_cov=show_cov, show_corr=show_corr)
    
    # Prepare data for summary plots
    shuffle_counts = np.array(n_shuffles_list)
    x_summary = shuffle_counts.copy().astype(float)
    x_summary[x_summary == 0] = 0.5 # For log scale visualization
    
    metrics = {'nn': ([], [], [], []), 'ent': ([], [], [], []), 'fro2': ([], [], [], [])}
    
    for n_shuffles in n_shuffles_list:
        for key, d_dict in zip(['corr'], [svs_corr_dict]):
            data = d_dict[n_shuffles]
            
            # Nuclear Norm (Sum of SVs)
            nn = np.sum(data, axis=1)
            metrics['nn'][0 if key=='cov' else 2].append(np.mean(nn))
            metrics['nn'][1 if key=='cov' else 3].append(np.std(nn))
            
            # Frobenius Norm Squared (Sum of Squares of SVs)
            fro2 = np.sum(data**2, axis=1)
            metrics['fro2'][0 if key=='cov' else 2].append(np.mean(fro2))
            metrics['fro2'][1 if key=='cov' else 3].append(np.std(fro2))
            
            # Entropy
            p = data / (np.sum(data, axis=1, keepdims=True) + 1e-12)
            log_p = np.zeros_like(p)
            log_p[p > 0] = np.log(p[p > 0])
            ent = -np.sum(p * log_p, axis=1)
            metrics['ent'][0 if key=='cov' else 2].append(np.mean(ent))
            metrics['ent'][1 if key=='cov' else 3].append(np.std(ent))
            
    # 3. Nuclear Norm vs Shuffles
    plot_summary_metric(x_summary, 
                        np.array(metrics['nn'][0]), np.array(metrics['nn'][1]),
                        np.array(metrics['nn'][2]), np.array(metrics['nn'][3]),
                        f"{title_prefix}: Nuclear Norm vs. Shuffles", "Sum of SVs $\sum \sigma_k$", 
                        "cov_svd_nuclear_norm", "crimson", "darkred", output_dir, show_cov=show_cov, show_corr=show_corr)
    
    # 4. Nuclear Norm Aligned (Double Y plot) - Only if both are shown
    if show_cov and show_corr:
        plot_metrics_comparison(x_summary,
                                np.array(metrics['nn'][0]), np.array(metrics['nn'][1]),
                                np.array(metrics['nn'][2]), np.array(metrics['nn'][3]),
                                f"{title_prefix}: Aligned Nuclear Norm Comparison", 
                                "Covariance Sum of SVs", "Correlation Sum of SVs",
                                "Covariance", "Correlation",
                                "tab:red", "tab:blue",
                                "cov_svd_nuclear_norm_aligned", output_dir)
    
    # 5. SV Entropy vs Shuffles
    plot_summary_metric(x_summary, 
                        np.array(metrics['ent'][0]), np.array(metrics['ent'][1]),
                        np.array(metrics['ent'][2]), np.array(metrics['ent'][3]),
                        f"{title_prefix}: Singular Value Entropy vs. Shuffles", "Entropy $H(\sigma)$", 
                        "cov_svd_entropy", "teal", "darkgreen", output_dir, show_cov=show_cov, show_corr=show_corr)
    
    # 6. Frobenius Norm Squared vs Shuffles
    plot_summary_metric(x_summary, 
                        np.array(metrics['fro2'][0]), np.array(metrics['fro2'][1]),
                        np.array(metrics['fro2'][2]), np.array(metrics['fro2'][3]),
                        f"{title_prefix}: Frobenius Norm Squared vs. Shuffles", "Sum of SVs Squared $\sum \sigma_k^2$", 
                        "cov_svd_frobenius_norm", "orange", "darkorange", output_dir, show_cov=show_cov, show_corr=show_corr)
    
    # 7. Frobenius Norm Squared Aligned (Double Y plot) - Only if both are shown
    if show_cov and show_corr:
        plot_metrics_comparison(x_summary,
                                np.array(metrics['fro2'][0]), np.array(metrics['fro2'][1]),
                                np.array(metrics['fro2'][2]), np.array(metrics['fro2'][3]),
                                f"{title_prefix}: Aligned Frobenius Norm Squared Comparison", 
                                "Covariance $\sum \sigma_k^2$", "Correlation $\sum \sigma_k^2$",
                                "Covariance", "Correlation",
                                "tab:red", "tab:blue",
                                "cov_svd_frobenius_norm_aligned", output_dir)
    
    # 8. Nuclear vs Frobenius Comparison (Double Y) - Per data type
    if show_cov:
        plot_metrics_comparison(x_summary,
                                np.array(metrics['nn'][0]), np.array(metrics['nn'][1]),
                                np.array(metrics['fro2'][0]), np.array(metrics['fro2'][1]),
                                f"{title_prefix}: Covariance Norms Comparison",
                                "Nuclear Norm $\sum \sigma_k$", "Squared Frobenius $\sum \sigma_k^2$",
                                "Nuclear Norm", "Squared Frobenius",
                                "crimson", "orange",
                                "cov_svd_norms_comparison_cov", output_dir)
    if show_corr:
        plot_metrics_comparison(x_summary,
                                np.array(metrics['nn'][2]), np.array(metrics['nn'][3]),
                                np.array(metrics['fro2'][2]), np.array(metrics['fro2'][3]),
                                f"{title_prefix}: Correlation Norms Comparison",
                                "Nuclear Norm $\sum \sigma_k$", "Squared Frobenius $\sum \sigma_k^2$",
                                "Nuclear Norm", "Squared Frobenius",
                                "darkred", "darkorange",
                                "cov_svd_norms_comparison_corr", output_dir)
        
        # 8b. Polished Correlation Norms Comparison
        plot_polished_norms_comparison(x_summary,
                                       np.array(metrics['nn'][2]), np.array(metrics['nn'][3]),
                                       np.array(metrics['fro2'][2]), np.array(metrics['fro2'][3]),
                                       f"{title_prefix}: Correlation Norms Comparison", output_dir)
    
    # 9. Cumulative Spectrum (Full & Top 10%)
    plot_cumulative_spectrum(n_shuffles_list, svs_cov_dict, svs_corr_dict, output_dir, title_prefix, percentile=None, show_cov=show_cov, show_corr=show_corr)
    plot_cumulative_spectrum(n_shuffles_list, svs_cov_dict, svs_corr_dict, output_dir, title_prefix, percentile=0.1, show_cov=show_cov, show_corr=show_corr)
    
    print(f"All plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot SVD analysis results.")
    parser.add_argument("dataset", type=str, help="Dataset name")
    parser.add_argument("view1_idx", type=int, nargs='?', default=None, help="Index of view 1")
    parser.add_argument("view2_idx", type=int, nargs='?', default=None, help="Index of view 2")
    parser.add_argument("--cov", action="store_true", help="Plot covariance metrics")
    parser.add_argument("--corr", action="store_true", help="Plot correlation metrics")
    parser.add_argument("--n_repeats", type=int, default=None, help="Ignored (for compatibility with cov_svd.py args)")
    
    args = parser.parse_args()
    plot_analysis(args)
