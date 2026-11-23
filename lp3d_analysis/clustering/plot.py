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




def plot_cluster_radar_profiles(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Optional[List[str]] = None,
    title: str = "Cluster Feature Profiles",
    figsize_per_cluster: Tuple[float, float] = (4, 4),
    max_cols: int = 4,
    ylim: Tuple[float, float] = (-2.5, 2.5),
    standardize: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
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
        figsize: if provided, overrides figsize_per_cluster calculation
        save_path: if provided, save figure as PDF to this path
        
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
    
    # Determine figure size
    if figsize is not None:
        final_figsize = figsize
    else:
        final_figsize = (figsize_per_cluster[0] * cols, figsize_per_cluster[1] * rows)
    
    fig, axes = plt.subplots(
        rows, cols, 
        figsize=final_figsize,
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
    
    # Save figure if path provided
    if save_path:
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"✓ Radar plot saved to: {save_path}")
    
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




import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from typing import Tuple

def plot_enhanced_dendrogram(
    linkage_matrix: np.ndarray,
    title: str = "Hierarchical Clustering Dendrogram",
    truncate_mode: str = 'lastp',
    p: int = 30,
    figsize: Tuple[int, int] = (14, 6),
    color_threshold: float = None,
    node_size: int = 40,
):
    """
    Plot enhanced dendrogram from linkage matrix with visible nodes (circles).
    
    Args:
        linkage_matrix: scipy linkage matrix (can be None if not computed)
        title: plot title
        truncate_mode: 'lastp' shows last p merged clusters
        p: number of clusters to show in truncated dendrogram
        figsize: figure size
        color_threshold: threshold for coloring dendrogram branches
        node_size: size of the circles used for nodes
        
    Returns:
        fig, ax: matplotlib figure and axis (or None, None if linkage_matrix is None)
    """
    if linkage_matrix is None:
        print("⚠ Linkage matrix is None - dendrogram cannot be plotted")
        print("  (This happens when dataset is too large for linkage computation)")
        return None, None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 1. Plot the standard dendrogram and capture its output dictionary
    ddata = dendrogram(
        linkage_matrix,
        truncate_mode=truncate_mode,
        p=p,
        ax=ax,
        color_threshold=color_threshold,
        above_threshold_color='gray',
    )
    
    # Extract the line coordinates and colors
    icoord = np.array(ddata['icoord'])
    dcoord = np.array(ddata['dcoord'])
    colors = ddata['color_list']
    
    # 2. Overlay circles at every structural point
    for x_coords, y_coords, color in zip(icoord, dcoord, colors):
        # x_coords and y_coords define the upside-down 'U' shape of the branch
        # Indices: 0=bottom-left, 1=top-left, 2=top-right, 3=bottom-right
        
        # Draw smaller circles at the bottom points (leaves or sub-clusters)
        ax.scatter([x_coords[0], x_coords[3]], [y_coords[0], y_coords[3]], 
                   color=color, s=node_size, zorder=3, edgecolors='white', linewidths=1)
        
        # Calculate the midpoint for the top junction and draw a slightly larger circle
        mid_x = 0.5 * (x_coords[1] + x_coords[2])
        top_y = y_coords[1] 
        ax.scatter([mid_x], [top_y], 
                   color=color, s=node_size * 1.5, zorder=3, edgecolors='white', linewidths=1)
    
    # 3. Clean up the aesthetics
    ax.set_title(title, fontsize=16, pad=15)
    ax.set_xlabel('Sample Index (or cluster size)', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    
    # Remove top and right borders for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add a subtle grid to easily trace distances across the chart
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig, ax


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from typing import Tuple, Optional, Dict, List


def linkage_to_digraph(linkage_matrix: np.ndarray) -> Tuple[nx.DiGraph, int]:
    n_leaves = linkage_matrix.shape[0] + 1
    G = nx.DiGraph()
    for i, (left, right, dist, count) in enumerate(linkage_matrix):
        parent = n_leaves + i
        left, right = int(left), int(right)
        G.add_edge(parent, left)
        G.add_edge(parent, right)
        G.nodes[parent]['dist'] = dist
        G.nodes[parent]['count'] = int(count)
    root = n_leaves + len(linkage_matrix) - 1
    return G, root


def hierarchy_pos_horizontal(
    G: nx.DiGraph,
    root: int,
    y_spread: float = 1.0,
    x_gap: float = 1.0,
) -> Dict[int, Tuple[float, float]]:
    """
    Leaf-count-aware horizontal layout (root on LEFT, leaves on RIGHT).
    
    - x-axis = depth (each level is x_gap apart)
    - y-axis = spread proportional to number of leaves below each subtree
    
    This prevents the 'collapsed to one corner' problem by giving each
    subtree vertical space proportional to how many leaves it contains.
    """

    # Count leaves below each node
    def count_leaves(node):
        children = list(G.neighbors(node))
        if not children:
            return 1
        return sum(count_leaves(c) for c in children)

    leaf_counts = {n: count_leaves(n) for n in G.nodes()}

    pos = {}

    def assign_pos(node, x, y_min, y_max):
        y_center = (y_min + y_max) / 2
        pos[node] = (x, y_center)

        children = list(G.neighbors(node))
        if not children:
            return

        total_leaves = sum(leaf_counts[c] for c in children)
        y_cursor = y_min
        for child in children:
            fraction = leaf_counts[child] / total_leaves
            child_y_min = y_cursor
            child_y_max = y_cursor + fraction * (y_max - y_min)
            assign_pos(child, x + x_gap, child_y_min, child_y_max)
            y_cursor = child_y_max

    assign_pos(root, x=0, y_min=0, y_max=y_spread)
    return pos


# def plot_dendrogram(
#     linkage_matrix: Optional[np.ndarray],
#     title: str = "Hierarchical representation of behavior",
#     figsize: Tuple[int, int] = (16, 9),
#     leaf_node_size: int = 1200,
#     internal_node_size: int = 200,
#     leaf_node_color: str = "#4a4a4a",
#     internal_node_color: str = "white",
#     edge_color: str = "#222222",
#     font_color: str = "white",
#     font_size: int = 9,
#     edge_width: float = 1.8,
#     cluster_communities: Optional[Dict[int, List[int]]] = None,
#     community_colors: Optional[Dict[int, str]] = None,
# ) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:

#     if linkage_matrix is None:
#         print("⚠ Linkage matrix is None — cannot plot.")
#         return None, None

#     G, root = linkage_to_digraph(linkage_matrix)
#     n_leaves = linkage_matrix.shape[0] + 1
#     leaf_nodes = list(range(n_leaves))
#     internal_nodes = list(range(n_leaves, n_leaves + len(linkage_matrix)))

#     # Use leaf-count-aware horizontal layout
#     pos = hierarchy_pos_horizontal(G, root, y_spread=1.0, x_gap=1.0)

#     fig, ax = plt.subplots(figsize=figsize)
#     ax.set_title(title, fontsize=13, fontweight='normal', pad=16)
#     ax.axis('off')

#     # Draw edges
#     nx.draw_networkx_edges(
#         G, pos, ax=ax,
#         edge_color=edge_color,
#         width=edge_width,
#         arrows=False,
#     )

#     # Internal nodes: small white circles with border
#     nx.draw_networkx_nodes(
#         G, pos,
#         nodelist=internal_nodes,
#         ax=ax,
#         node_size=internal_node_size,
#         node_color=internal_node_color,
#         edgecolors=edge_color,
#         linewidths=1.5,
#     )

#     # Leaf nodes: filled dark circles, all same size
#     nx.draw_networkx_nodes(
#         G, pos,
#         nodelist=leaf_nodes,
#         ax=ax,
#         node_size=leaf_node_size,
#         node_color=leaf_node_color,
#         edgecolors='none',
#     )

#     # Labels inside leaf nodes
#     nx.draw_networkx_labels(
#         G, pos,
#         labels={n: str(n) for n in leaf_nodes},
#         ax=ax,
#         font_size=font_size,
#         font_color=font_color,
#         font_weight='bold',
#     )

#     # Optional community halos
#     if cluster_communities:
#         _draw_community_halos(ax, pos, cluster_communities, community_colors)

#     # Flip x-axis so root is on LEFT and leaves fan out to the RIGHT
#     # (matches VAME figure orientation)
#     ax.invert_xaxis()

#     plt.tight_layout()
#     return fig, ax


# def _draw_community_halos(
#     ax, pos, cluster_communities, community_colors=None, padding=0.06, alpha=0.25,
# ):
#     default_colors = [
#         '#e6194b', '#f58231', '#3cb44b', '#4363d8',
#         '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#aaaaaa',
#     ]
#     for i, (comm_id, nodes) in enumerate(cluster_communities.items()):
#         xs = [pos[n][0] for n in nodes if n in pos]
#         ys = [pos[n][1] for n in nodes if n in pos]
#         if not xs:
#             continue
#         cx = (max(xs) + min(xs)) / 2
#         cy = (max(ys) + min(ys)) / 2
#         w = max((max(xs) - min(xs)) + padding * 2, padding * 3)
#         h = max((max(ys) - min(ys)) + padding * 2, padding * 3)
#         color = (community_colors or {}).get(comm_id, default_colors[i % len(default_colors)])
#         ax.add_patch(mpatches.Ellipse(
#             (cx, cy), width=w, height=h,
#             color=color, alpha=alpha, zorder=0,
#             transform=ax.transData,
#         ))

from typing import Optional, Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches

def plot_dendrogram(
    linkage_matrix: Optional[np.ndarray],
    title: str = "Hierarchical representation of behavior",
    figsize: Tuple[int, int] = (10, 14),  # <- taller for vertical
    leaf_node_size: int = 1200,
    internal_node_size: int = 200,
    leaf_node_color: str = "#4a4a4a",
    internal_node_color: str = "white",
    edge_color: str = "#222222",
    font_color: str = "white",
    font_size: int = 9,
    edge_width: float = 1.8,
    cluster_communities: Optional[Dict[int, List[int]]] = None,
    community_colors: Optional[Dict[int, str]] = None,
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:

    if linkage_matrix is None:
        print("⚠ Linkage matrix is None — cannot plot.")
        return None, None

    G, root = linkage_to_digraph(linkage_matrix)
    n_leaves = linkage_matrix.shape[0] + 1
    leaf_nodes = list(range(n_leaves))
    internal_nodes = list(range(n_leaves, n_leaves + len(linkage_matrix)))

    # --- Layout: compute horizontal positions, then rotate to vertical ---
    pos_h = hierarchy_pos_horizontal(G, root, y_spread=1.0, x_gap=1.0)

    # Rotate 90° clockwise: (x, y) -> (y, -x)
    pos = {n: (y, x) for n, (x, y) in pos_h.items()}

    # Flip vertically so root is at TOP and leaves go DOWN
    ys = [p[1] for p in pos.values()]
    ymin, ymax = min(ys), max(ys)
    pos = {n: (x, ymax - (y - ymin)) for n, (x, y) in pos.items()}
    # -------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=13, fontweight='normal', pad=16)
    ax.axis('off')

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_color,
        width=edge_width,
        arrows=False,
    )

    # Internal nodes: small white circles with border
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=internal_nodes,
        ax=ax,
        node_size=internal_node_size,
        node_color=internal_node_color,
        edgecolors=edge_color,
        linewidths=1.5,
    )

    # Leaf nodes: filled dark circles
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=leaf_nodes,
        ax=ax,
        node_size=leaf_node_size,
        node_color=leaf_node_color,
        edgecolors='none',
    )

    # Labels inside leaf nodes
    nx.draw_networkx_labels(
        G, pos,
        labels={n: str(n) for n in leaf_nodes},
        ax=ax,
        font_size=font_size,
        font_color=font_color,
        font_weight='bold',
    )

    # Optional community halos
    if cluster_communities:
        _draw_community_halos(ax, pos, cluster_communities, community_colors)

    # NOTE: removed ax.invert_xaxis() because we are vertical now

    plt.tight_layout()
    return fig, ax


def _draw_community_halos(
    ax, pos, cluster_communities, community_colors=None, padding=0.06, alpha=0.25,
):
    default_colors = [
        '#e6194b', '#f58231', '#3cb44b', '#4363d8',
        '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#aaaaaa',
    ]
    for i, (comm_id, nodes) in enumerate(cluster_communities.items()):
        xs = [pos[n][0] for n in nodes if n in pos]
        ys = [pos[n][1] for n in nodes if n in pos]
        if not xs:
            continue
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        w = max((max(xs) - min(xs)) + padding * 2, padding * 3)
        h = max((max(ys) - min(ys)) + padding * 2, padding * 3)
        color = (community_colors or {}).get(comm_id, default_colors[i % len(default_colors)])
        ax.add_patch(mpatches.Ellipse(
            (cx, cy), width=w, height=h,
            color=color, alpha=alpha, zorder=0,
            transform=ax.transData,
        ))