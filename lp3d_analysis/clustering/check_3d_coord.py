# ============================================================
# 3D SPACE COMPARISON ACROSS SESSIONS
# ============================================================
import pandas as pd
from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

def compare_3d_space_across_sessions(
    all_3d_data: Dict,
    model_name: str,
    keypoints: List[str],
    figsize: Tuple[int, int] = (18, 12),
):
    """
    Visualize and compare 3D coordinate distributions across sessions.
    
    Args:
        all_3d_data: dict[model][session] -> (n_frames, n_keypoints, 3)
        model_name: which model to analyze
        keypoints: list of keypoint names
    """
    sessions = list(all_3d_data[model_name].keys())
    n_sessions = len(sessions)
    n_keypoints = len(keypoints)
    # re
    
    # Short session names for plotting
    short_names = {s: s.split('.')[-1][:8] for s in sessions}
    
    # Colors for sessions
    colors = plt.cm.tab10(np.linspace(0, 1, n_sessions))
    
    # ============================================================
    # PLOT 1: 3D scatter of all keypoints per session
    # ============================================================
    fig = plt.figure(figsize=figsize)
    
    for kp_idx, kp_name in enumerate(keypoints):
        ax = fig.add_subplot(2, n_keypoints, kp_idx + 1, projection='3d')
        
        for sess_idx, session_name in enumerate(sessions):
            pts = all_3d_data[model_name][session_name][:, kp_idx, :]  # (T, 3)
            
            # Subsample for visualization
            n_plot = min(2000, len(pts))
            idx = np.random.choice(len(pts), n_plot, replace=False)
            pts_sub = pts[idx]
            
            # Remove NaN
            valid = ~np.any(np.isnan(pts_sub), axis=1)
            pts_valid = pts_sub[valid]
            
            ax.scatter(pts_valid[:, 0], pts_valid[:, 1], pts_valid[:, 2],
                      c=[colors[sess_idx]], s=1, alpha=0.3,
                      label=short_names[session_name])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{kp_name} - 3D positions')
        if kp_idx == 0:
            ax.legend(fontsize=7, loc='upper left')
    
    # ============================================================
    # PLOT 2: Distribution histograms per axis
    # ============================================================
    for kp_idx, kp_name in enumerate(keypoints):
        ax = fig.add_subplot(2, n_keypoints, n_keypoints + kp_idx + 1)
        
        for sess_idx, session_name in enumerate(sessions):
            pts = all_3d_data[model_name][session_name][:, kp_idx, :]
            
            # Plot X distribution (could do Y, Z separately)
            x_vals = pts[:, 0][~np.isnan(pts[:, 0])]
            ax.hist(x_vals, bins=50, alpha=0.4, color=colors[sess_idx],
                   label=short_names[session_name], density=True)
        
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Density')
        ax.set_title(f'{kp_name} - X distribution')
        ax.legend(fontsize=7)
    
    fig.suptitle(f'3D Space Comparison Across Sessions (Raw, Before Normalization)', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def compute_session_statistics(
    all_3d_data: Dict,
    model_name: str,
    keypoints: List[str],
) -> pd.DataFrame:
    """
    Compute summary statistics of 3D coordinates per session.
    
    Returns DataFrame with min, max, mean, std for each session/keypoint/axis.
    """
    results = []
    
    for session_name, pts_3d in all_3d_data[model_name].items():
        short_name = session_name.split('.')[-1][:8]
        
        for kp_idx, kp_name in enumerate(keypoints):
            pts = pts_3d[:, kp_idx, :]  # (T, 3)
            
            for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
                vals = pts[:, axis_idx]
                valid_vals = vals[~np.isnan(vals)]
                
                if len(valid_vals) > 0:
                    results.append({
                        'session': short_name,
                        'session_full': session_name,
                        'keypoint': kp_name,
                        'axis': axis_name,
                        'min': np.min(valid_vals),
                        'max': np.max(valid_vals),
                        'mean': np.mean(valid_vals),
                        'std': np.std(valid_vals),
                        'range': np.max(valid_vals) - np.min(valid_vals),
                        'n_valid': len(valid_vals),
                    })
    
    return pd.DataFrame(results)


def plot_session_ranges(
    df_stats: pd.DataFrame,
    keypoints: List[str],
    figsize: Tuple[int, int] = (14, 10),
):
    """
    Plot min/max ranges per session to visualize calibration differences.
    """
    sessions = df_stats['session'].unique()
    n_sessions = len(sessions)
    
    fig, axes = plt.subplots(len(keypoints), 3, figsize=figsize, sharey='row')
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_sessions))
    
    for row_idx, kp_name in enumerate(keypoints):
        for col_idx, axis_name in enumerate(['x', 'y', 'z']):
            ax = axes[row_idx, col_idx] if len(keypoints) > 1 else axes[col_idx]
            
            df_sub = df_stats[(df_stats['keypoint'] == kp_name) & 
                             (df_stats['axis'] == axis_name)]
            
            for sess_idx, session in enumerate(sessions):
                row = df_sub[df_sub['session'] == session]
                if len(row) == 0:
                    continue
                row = row.iloc[0]
                
                # Plot range as error bar
                ax.errorbar(sess_idx, row['mean'], 
                           yerr=[[row['mean'] - row['min']], [row['max'] - row['mean']]],
                           fmt='o', color=colors[sess_idx], capsize=5, capthick=2,
                           markersize=8, label=session if row_idx == 0 and col_idx == 0 else None)
            
            ax.set_xticks(range(n_sessions))
            ax.set_xticklabels(sessions, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel(f'{axis_name.upper()} (units)')
            ax.set_title(f'{kp_name} - {axis_name.upper()}')
            ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('3D Coordinate Ranges by Session\n(Error bars show min-max range)', 
                fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def quantify_session_differences(
    all_3d_data: Dict,
    model_name: str,
    keypoints: List[str],
) -> pd.DataFrame:
    """
    Quantify pairwise differences between sessions using statistical tests.
    
    Returns DataFrame with KS-test statistics between session pairs.
    """
    sessions = list(all_3d_data[model_name].keys())
    results = []
    
    for kp_idx, kp_name in enumerate(keypoints):
        for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
            for i, sess1 in enumerate(sessions):
                for j, sess2 in enumerate(sessions):
                    if i >= j:
                        continue
                    
                    pts1 = all_3d_data[model_name][sess1][:, kp_idx, axis_idx]
                    pts2 = all_3d_data[model_name][sess2][:, kp_idx, axis_idx]
                    
                    # Remove NaN
                    pts1 = pts1[~np.isnan(pts1)]
                    pts2 = pts2[~np.isnan(pts2)]
                    
                    if len(pts1) < 100 or len(pts2) < 100:
                        continue
                    
                    # KS test
                    ks_stat, ks_pval = stats.ks_2samp(pts1, pts2)
                    
                    # Mean difference
                    mean_diff = np.abs(np.mean(pts1) - np.mean(pts2))
                    
                    # Range overlap
                    range1 = (np.min(pts1), np.max(pts1))
                    range2 = (np.min(pts2), np.max(pts2))
                    overlap = max(0, min(range1[1], range2[1]) - max(range1[0], range2[0]))
                    total_range = max(range1[1], range2[1]) - min(range1[0], range2[0])
                    overlap_pct = 100 * overlap / total_range if total_range > 0 else 0
                    
                    results.append({
                        'keypoint': kp_name,
                        'axis': axis_name,
                        'session1': sess1.split('.')[-1][:8],
                        'session2': sess2.split('.')[-1][:8],
                        'ks_statistic': ks_stat,
                        'ks_pvalue': ks_pval,
                        'mean_diff': mean_diff,
                        'range_overlap_pct': overlap_pct,
                    })
    
    return pd.DataFrame(results)


def plot_pairwise_distributions(
    all_3d_data: Dict,
    model_name: str,
    keypoints: List[str],
    axis: str = 'x',
    figsize: Tuple[int, int] = (16, 12),
):
    """
    Plot pairwise distribution comparisons between sessions.
    """
    sessions = list(all_3d_data[model_name].keys())
    n_sessions = len(sessions)
    short_names = {s: s.split('.')[-1][:8] for s in sessions}
    
    fig, axes = plt.subplots(n_sessions, n_sessions, figsize=figsize)
    
    axis_idx = ['x', 'y', 'z'].index(axis)
    
    for i, sess1 in enumerate(sessions):
        for j, sess2 in enumerate(sessions):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: show single session histogram
                for kp_idx, kp_name in enumerate(keypoints):
                    pts = all_3d_data[model_name][sess1][:, kp_idx, axis_idx]
                    pts = pts[~np.isnan(pts)]
                    ax.hist(pts, bins=30, alpha=0.5, label=kp_name, density=True)
                ax.set_title(short_names[sess1], fontsize=9)
                if i == 0 and j == 0:
                    ax.legend(fontsize=6)
            elif i < j:
                # Upper triangle: scatter plot comparison (pawL only for clarity)
                kp_idx = keypoints.index('pawL')
                pts1 = all_3d_data[model_name][sess1][:, kp_idx, axis_idx]
                pts2 = all_3d_data[model_name][sess2][:, kp_idx, axis_idx]
                
                # Subsample
                n = min(1000, len(pts1), len(pts2))
                ax.scatter(pts1[:n], pts2[:n], s=1, alpha=0.3)
                
                # Add diagonal line
                lims = [min(np.nanmin(pts1), np.nanmin(pts2)),
                       max(np.nanmax(pts1), np.nanmax(pts2))]
                ax.plot(lims, lims, 'r--', alpha=0.5)
                ax.set_xlabel(short_names[sess1], fontsize=7)
                ax.set_ylabel(short_names[sess2], fontsize=7)
            else:
                # Lower triangle: KS statistic heatmap value
                kp_idx = keypoints.index('pawL')
                pts1 = all_3d_data[model_name][sess1][:, kp_idx, axis_idx]
                pts2 = all_3d_data[model_name][sess2][:, kp_idx, axis_idx]
                pts1 = pts1[~np.isnan(pts1)]
                pts2 = pts2[~np.isnan(pts2)]
                
                if len(pts1) > 100 and len(pts2) > 100:
                    ks_stat, _ = stats.ks_2samp(pts1, pts2)
                    ax.text(0.5, 0.5, f'KS={ks_stat:.3f}', 
                           ha='center', va='center', fontsize=10,
                           transform=ax.transAxes)
                    ax.set_facecolor(plt.cm.RdYlGn_r(ks_stat))
                ax.set_xticks([])
                ax.set_yticks([])
    
    fig.suptitle(f'Pairwise Session Comparison ({axis.upper()} axis, pawL)\n'
                f'Upper: scatter | Diagonal: histogram | Lower: KS statistic',
                fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


# ============================================================
# QUICK SUMMARY FUNCTION
# ============================================================

def check_3d_space_consistency(
    all_3d_data: Dict,
    model_name: str,
    keypoints: List[str],
):
    """
    Quick check of 3D space consistency across sessions.
    Prints summary and generates key plots.
    """
    print("=" * 70)
    print("3D SPACE CONSISTENCY CHECK ACROSS SESSIONS")
    print("=" * 70)
    
    # 1. Compute statistics
    df_stats = compute_session_statistics(all_3d_data, model_name, keypoints)
    
    print("\n1. COORDINATE RANGES PER SESSION:")
    print("-" * 50)
    
    for kp in keypoints:
        print(f"\n  {kp}:")
        df_kp = df_stats[df_stats['keypoint'] == kp]
        
        for axis in ['x', 'y', 'z']:
            df_axis = df_kp[df_kp['axis'] == axis]
            ranges = df_axis[['session', 'min', 'max', 'range']].values
            
            min_range = df_axis['range'].min()
            max_range = df_axis['range'].max()
            range_ratio = max_range / min_range if min_range > 0 else np.inf
            
            print(f"    {axis.upper()}: range varies from {min_range:.1f} to {max_range:.1f} "
                  f"(ratio: {range_ratio:.2f}x)")
    
    # 2. Quantify differences
    print("\n2. PAIRWISE SESSION DIFFERENCES (KS test):")
    print("-" * 50)
    
    df_diff = quantify_session_differences(all_3d_data, model_name, keypoints)
    
    # Summary: average KS statistic per axis
    for axis in ['x', 'y', 'z']:
        df_axis = df_diff[df_diff['axis'] == axis]
        mean_ks = df_axis['ks_statistic'].mean()
        mean_overlap = df_axis['range_overlap_pct'].mean()
        print(f"  {axis.upper()}: mean KS={mean_ks:.3f}, mean range overlap={mean_overlap:.1f}%")
    
    # Flag problematic pairs
    high_ks = df_diff[df_diff['ks_statistic'] > 0.3]
    if len(high_ks) > 0:
        print(f"\n  WARNING: {len(high_ks)} session pairs have KS > 0.3 (significant difference)")
        print("  Most different pairs:")
        top_diff = high_ks.nlargest(5, 'ks_statistic')[['session1', 'session2', 'keypoint', 'axis', 'ks_statistic']]
        print(top_diff.to_string(index=False))
    
    # 3. Recommendation
    print("\n3. RECOMMENDATION:")
    print("-" * 50)
    
    mean_ks_overall = df_diff['ks_statistic'].mean()
    mean_overlap_overall = df_diff['range_overlap_pct'].mean()
    
    if mean_ks_overall < 0.1 and mean_overlap_overall > 80:
        print("  ✓ Sessions appear CONSISTENT - normalization may not be critical")
    elif mean_ks_overall < 0.2 and mean_overlap_overall > 60:
        print("  ~ Sessions show MODERATE differences - normalization RECOMMENDED")
    else:
        print("  ✗ Sessions show LARGE differences - normalization REQUIRED")
        print("    Consider per-session min-max scaling to [-1, 1]")
    
    # 4. Generate plots
    print("\n4. GENERATING PLOTS...")
    
    fig1 = compare_3d_space_across_sessions(all_3d_data, model_name, keypoints)
    plt.show()
    
    fig2 = plot_session_ranges(df_stats, keypoints)
    plt.show()
    
    fig3 = plot_pairwise_distributions(all_3d_data, model_name, keypoints, axis='x')
    plt.show()
    
    return df_stats, df_diff