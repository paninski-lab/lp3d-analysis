from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import pi
from scipy.cluster.hierarchy import dendrogram
from lp3d_analysis.clustering.utils import pool_features_all_models_all_sessions, split_labels_by_model, split_embeddings_by_model

# ============================================================
# UPDATED VISUALIZATION WITH MODEL COLORING
# ============================================================
def plot_embeddings_three_ways(
    embedding_umap: np.ndarray,
    embedding_tsne: np.ndarray,
    cluster_labels: np.ndarray,
    session_labels: List[str],
    model_labels: List[str],
    cluster_names: Dict[int, str],
    model_short_names: Dict[str, str],
    figsize: Tuple[int, int] = (20, 15),
):
    """
    Plot UMAP and t-SNE colored by: (1) cluster, (2) session, (3) model.
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    n_clusters = len(np.unique(cluster_labels))
    unique_sessions = list(set(session_labels))
    unique_models = list(set(model_labels))
    n_sessions = len(unique_sessions)
    n_models = len(unique_models)
    
    # Color maps
    cluster_cmap = plt.cm.tab10 if n_clusters <= 10 else plt.cm.tab20
    session_cmap = plt.cm.Set3 if n_sessions <= 12 else plt.cm.tab20
    model_cmap = plt.cm.Set1
    
    # Numeric mappings
    session_to_idx = {s: i for i, s in enumerate(unique_sessions)}
    session_numeric = np.array([session_to_idx[s] for s in session_labels])
    
    model_to_idx = {m: i for i, m in enumerate(unique_models)}
    model_numeric = np.array([model_to_idx[m] for m in model_labels])
    
    # Row 1: Color by CLUSTER
    for col, (embedding, name) in enumerate([(embedding_umap, 'UMAP'), (embedding_tsne, 't-SNE')]):
        ax = axes[0, col]
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                            c=cluster_labels, cmap=cluster_cmap, s=1, alpha=0.5)
        ax.set_title(f'{name} - Colored by Cluster', fontsize=12)
        ax.set_xlabel(f'{name} 1')
        ax.set_ylabel(f'{name} 2')
    
    # Row 2: Color by SESSION
    for col, (embedding, name) in enumerate([(embedding_umap, 'UMAP'), (embedding_tsne, 't-SNE')]):
        ax = axes[1, col]
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                            c=session_numeric, cmap=session_cmap, s=1, alpha=0.5)
        ax.set_title(f'{name} - Colored by Session', fontsize=12)
        ax.set_xlabel(f'{name} 1')
        ax.set_ylabel(f'{name} 2')
    
    # Row 3: Color by MODEL
    for col, (embedding, name) in enumerate([(embedding_umap, 'UMAP'), (embedding_tsne, 't-SNE')]):
        ax = axes[2, col]
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                            c=model_numeric, cmap=model_cmap, s=1, alpha=0.5)
        ax.set_title(f'{name} - Colored by Model', fontsize=12)
        ax.set_xlabel(f'{name} 1')
        ax.set_ylabel(f'{name} 2')
    
    # Add legends on the right side
    # Cluster legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=cluster_cmap(i/n_clusters),
                         markersize=8, label=cluster_names.get(i, f'C{i}'))
              for i in range(min(n_clusters, 10))]
    axes[0, 1].legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5),
                     fontsize=7, title='Clusters')
    
    # Session legend
    short_session_names = [s.split('.')[-1][:8] for s in unique_sessions]
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=session_cmap(i/n_sessions),
                         markersize=8, label=short_session_names[i])
              for i in range(n_sessions)]
    axes[1, 1].legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5),
                     fontsize=7, title='Sessions')
    
    # Model legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=model_cmap(i/n_models),
                         markersize=8, label=model_short_names.get(m, m))
              for i, m in enumerate(unique_models)]
    axes[2, 1].legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5),
                     fontsize=8, title='Models')
    
    fig.suptitle('Multi-Session, Multi-Model Clustering', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    title: str = "Hierarchical Clustering Dendrogram",
    truncate_mode: str = 'lastp',
    p: int = 30,
    figsize: Tuple[int, int] = (14, 6),
    color_threshold: float = None,
):
    """
    Plot dendrogram from linkage matrix.
    
    Args:
        linkage_matrix: scipy linkage matrix (can be None if not computed)
        title: plot title
        truncate_mode: 'lastp' shows last p merged clusters
        p: number of clusters to show in truncated dendrogram
        figsize: figure size
        color_threshold: threshold for coloring dendrogram branches
        
    Returns:
        fig, ax: matplotlib figure and axis (or None, None if linkage_matrix is None)
    """
    if linkage_matrix is None:
        print("⚠ Linkage matrix is None - dendrogram cannot be plotted")
        print("  (This happens when dataset is too large for linkage computation)")
        return None, None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    dendrogram(
        linkage_matrix,
        truncate_mode=truncate_mode,
        p=p,
        ax=ax,
        color_threshold=color_threshold,
        above_threshold_color='gray',
    )
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Sample Index (or cluster size)')
    ax.set_ylabel('Distance')
    
    plt.tight_layout()
    return fig, ax



def plot_cluster_radar_profiles(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Optional[List[str]] = None,
    title: str = "Cluster Feature Profiles",
    figsize_per_cluster: Tuple[float, float] = (4, 4),
    max_cols: int = 4,
    ylim: Tuple[float, float] = (-2.5, 2.5),
    standardize: bool = True,
) -> Tuple[plt.Figure, Dict[int, Dict]]:
    """
    Create radar (spider) plots showing feature profiles for ALL clusters.
    
    This helps understand what features characterize each cluster by showing
    the mean standardized feature values as a radar chart.
    
    Args:
        features: (n_samples, n_features) array - can be raw or scaled
        labels: (n_samples,) cluster assignments
        feature_names: list of feature names (auto-generated if None)
        title: main figure title
        figsize_per_cluster: (width, height) per subplot
        max_cols: maximum columns in subplot grid
        ylim: y-axis limits for radar plots (standardized units)
        standardize: if True, z-score normalize features before plotting
        
    Returns:
        fig: matplotlib figure
        cluster_profiles: dict mapping cluster_id -> {'mean': array, 'std': array, 'n': int}
    """
    # Get unique clusters (sorted)
    unique_clusters = np.sort(np.unique(labels))
    n_clusters = len(unique_clusters)
    n_features = features.shape[1]
    
    # Generate feature names if not provided
    if feature_names is None or len(feature_names) != n_features:
        feature_names = [f'F{i}' for i in range(n_features)]
    
    # Standardize features for fair comparison
    if standardize:
        feats_std = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    else:
        feats_std = features
    
    # Compute cluster profiles
    cluster_profiles = {}
    for c in unique_clusters:
        mask = labels == c
        cluster_profiles[c] = {
            'mean': feats_std[mask].mean(axis=0),
            'std': feats_std[mask].std(axis=0),
            'n': mask.sum()
        }
    
    # Setup radar chart angles
    angles = [n / float(n_features) * 2 * pi for n in range(n_features)]
    angles += angles[:1]  # close the polygon
    
    # Create subplot grid
    cols = min(max_cols, n_clusters)
    rows = int(np.ceil(n_clusters / cols))
    
    fig, axes = plt.subplots(
        rows, cols, 
        figsize=(figsize_per_cluster[0] * cols, figsize_per_cluster[1] * rows),
        subplot_kw=dict(polar=True)
    )
    
    # Handle single cluster case
    if n_clusters == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for idx, c in enumerate(unique_clusters):
        ax = axes[idx]
        data = cluster_profiles[c]
        
        # Prepare values (close the polygon)
        values = data['mean'].tolist()
        values += values[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[idx], markersize=4)
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
        
        # Feature labels (truncate long names)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([fn[:12] for fn in feature_names], size=7)
        
        # Title with cluster info
        pct = 100 * data['n'] / len(labels)
        ax.set_title(f'Cluster {c}\n(n={data["n"]:,}, {pct:.1f}%)', 
                     size=10, fontweight='bold', pad=10)
        ax.set_ylim(ylim)
        
        # Add gridlines
        ax.set_rticks([-2, -1, 0, 1, 2])
        ax.set_yticklabels(['-2', '-1', '0', '1', '2'], size=6, color='gray')
    
    # Hide unused subplots
    for idx in range(n_clusters, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig, cluster_profiles



def plot_reprojection_by_cluster(
    df_errors: pd.DataFrame,
    cluster_names: Dict[int, str],
    figsize: Tuple[int, int] = (14, 6),
    use_sem: bool = True,  # New parameter to toggle SEM error bars
):
    """
    Plot grouped bar charts of median reprojection error by model and cluster.
    Now includes SEM error bars across sessions.
    """
    keypoints = df_errors['keypoint'].unique()
    models = df_errors['model'].unique()
    n_clusters = df_errors['cluster'].nunique()
    
    fig, axes = plt.subplots(1, len(keypoints), figsize=figsize)
    if len(keypoints) == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    bar_width = 0.8 / len(models)
    
    for ax, kp in zip(axes, keypoints):
        df_kp = df_errors[df_errors['keypoint'] == kp]
        
        x = np.arange(n_clusters)
        
        for i, model in enumerate(models):
            df_model = df_kp[df_kp['model'] == model].sort_values('cluster')
            errors = df_model['median_error'].values
            
            # Get SEM values if available
            if use_sem and 'sem_error' in df_model.columns:
                sem_values = df_model['sem_error'].values
                # Replace NaN with 0 for plotting
                sem_values = np.nan_to_num(sem_values, nan=0.0)
            else:
                sem_values = None
            
            offset = (i - len(models)/2 + 0.5) * bar_width
            ax.bar(
                x + offset, 
                errors, 
                bar_width, 
                label=model, 
                color=colors[i], 
                alpha=0.8,
                yerr=sem_values,  # Add error bars
                capsize=3,  # Add caps to error bars
                error_kw={'elinewidth': 1, 'capthick': 1}
            )
        
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Median Reprojection Error (pixels)')
        ax.set_title(f'{kp}')
        ax.set_xticks(x)
        
        xlabels = [cluster_names.get(i, f'C{i}').split('_')[0] for i in range(n_clusters)]
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Median Reprojection Error by Model and Cluster (± SEM)', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig




# def plot_reprojection_boxplots(
#     reproj_errors: Dict,
#     cluster_labels: np.ndarray,
#     valid_mask: np.ndarray,
#     keypoints: List[str],
#     models_of_interest: List[str],
#     model_short_names: Dict[str, str],
#     cluster_names: Dict[int, str],
#     keypoints_of_interest: List[str] = ['pawL', 'pawR'],
#     figsize: Tuple[int, int] = (16, 8),
# ):
#     """
#     Plot boxplots of reprojection error distribution by cluster.
#     """
#     kp_indices = {kp: keypoints.index(kp) for kp in keypoints_of_interest}
#     n_clusters = len(np.unique(cluster_labels))
    
#     fig, axes = plt.subplots(len(keypoints_of_interest), n_clusters, 
#                             figsize=figsize, sharey='row')
    
#     for row_idx, kp_name in enumerate(keypoints_of_interest):
#         kp_idx = kp_indices[kp_name]
        
#         for cluster_id in range(n_clusters):
#             ax = axes[row_idx, cluster_id] if len(keypoints_of_interest) > 1 else axes[cluster_id]
            
#             cluster_mask = cluster_labels == cluster_id
            
#             data_to_plot = []
#             labels = []
            
#             for model_name in models_of_interest:
#                 if model_name not in reproj_errors:
#                     continue
                
#                 # Pool errors
#                 all_errors = []
#                 for session_name in reproj_errors[model_name].keys():
#                     errors = reproj_errors[model_name][session_name]
#                     all_errors.append(errors)
                
#                 if not all_errors:
#                     continue
                
#                 pooled_errors = np.concatenate(all_errors, axis=0)
#                 pooled_errors_valid = pooled_errors[valid_mask]
                
#                 kp_errors = pooled_errors_valid[cluster_mask, kp_idx, :]
#                 kp_errors_mean = np.nanmean(kp_errors, axis=1)
#                 kp_errors_clean = kp_errors_mean[~np.isnan(kp_errors_mean)]
                
#                 data_to_plot.append(kp_errors_clean)
#                 labels.append(model_short_names.get(model_name, model_name))
            
#             if data_to_plot:
#                 bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
#                 colors = plt.cm.Set2(np.linspace(0, 1, len(data_to_plot)))
#                 for patch, color in zip(bp['boxes'], colors):
#                     patch.set_facecolor(color)
#                     patch.set_alpha(0.7)
            
#             cluster_label = cluster_names.get(cluster_id, f'C{cluster_id}')
#             ax.set_title(f'{cluster_label}', fontsize=9)
#             ax.tick_params(axis='x', rotation=45)
#             ax.grid(axis='y', alpha=0.3)
            
#             if cluster_id == 0:
#                 ax.set_ylabel(f'{kp_name}\nReproj. Error (px)')
    
#     fig.suptitle('Reprojection Error Distribution by Cluster', fontsize=14, y=1.02)
#     plt.tight_layout()
#     return fig