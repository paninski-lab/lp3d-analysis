import numpy as np
from typing import Dict, Tuple, List
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



# ============================================================
# PART 4: UMAP & t-SNE VISUALIZATION
# ============================================================

def compute_embeddings(
    features_scaled: np.ndarray,
    umap_params: Dict = None,
    tsne_params: Dict = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute both UMAP and t-SNE embeddings.
    """
    if umap_params is None:
        umap_params = {
            'n_components': 2,
            'n_neighbors': 199,
            'min_dist': 0.3,
            'random_state': 42,
        }
    
    if tsne_params is None:
        tsne_params = {
            'n_components': 2,
            'perplexity': 30,
            'random_state': 42,
            'n_iter': 1000,
        }
    
    print("Computing UMAP embedding...")
    reducer_umap = umap.UMAP(**umap_params)
    embedding_umap = reducer_umap.fit_transform(features_scaled)
    
    print("Computing t-SNE embedding...")
    reducer_tsne = TSNE(**tsne_params)
    embedding_tsne = reducer_tsne.fit_transform(features_scaled)
    
    return embedding_umap, embedding_tsne


def plot_embeddings_side_by_side(
    embedding_umap: np.ndarray,
    embedding_tsne: np.ndarray,
    labels: np.ndarray,
    session_labels: List[str],
    cluster_names: Dict[int, str] = None,
    title: str = "Embedding Visualization",
    figsize: Tuple[int, int] = (20, 10),
):
    """
    Plot UMAP and t-SNE side by side, colored by cluster and session.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    n_clusters = len(np.unique(labels))
    unique_sessions = list(set(session_labels))
    n_sessions = len(unique_sessions)
    
    # Color maps
    cluster_cmap = plt.cm.tab10 if n_clusters <= 10 else plt.cm.tab20
    session_cmap = plt.cm.Set3 if n_sessions <= 12 else plt.cm.tab20
    
    # Session to numeric mapping
    session_to_idx = {s: i for i, s in enumerate(unique_sessions)}
    session_numeric = np.array([session_to_idx[s] for s in session_labels])
    
    # Row 1: Color by cluster
    # UMAP
    ax = axes[0, 0]
    scatter = ax.scatter(embedding_umap[:, 0], embedding_umap[:, 1],
                        c=labels, cmap=cluster_cmap, s=1, alpha=0.5)
    ax.set_title('UMAP - Colored by Cluster', fontsize=12)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    # t-SNE
    ax = axes[0, 1]
    scatter = ax.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1],
                        c=labels, cmap=cluster_cmap, s=1, alpha=0.5)
    ax.set_title('t-SNE - Colored by Cluster', fontsize=12)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # Add cluster legend
    if cluster_names:
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=cluster_cmap(i/n_clusters), 
                             markersize=8, label=cluster_names.get(i, f'C{i}'))
                  for i in range(n_clusters)]
        axes[0, 1].legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5),
                         fontsize=8, title='Clusters')
    
    # Row 2: Color by session
    # UMAP
    ax = axes[1, 0]
    scatter = ax.scatter(embedding_umap[:, 0], embedding_umap[:, 1],
                        c=session_numeric, cmap=session_cmap, s=1, alpha=0.5)
    ax.set_title('UMAP - Colored by Session', fontsize=12)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    # t-SNE
    ax = axes[1, 1]
    scatter = ax.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1],
                        c=session_numeric, cmap=session_cmap, s=1, alpha=0.5)
    ax.set_title('t-SNE - Colored by Session', fontsize=12)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # Add session legend
    short_session_names = [s.split('.')[-1][:8] for s in unique_sessions]
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=session_cmap(i/n_sessions),
                         markersize=8, label=short_session_names[i])
              for i in range(n_sessions)]
    axes[1, 1].legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5),
                     fontsize=8, title='Sessions')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


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
