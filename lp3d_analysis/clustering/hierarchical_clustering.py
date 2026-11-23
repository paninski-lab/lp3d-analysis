import numpy as np
from typing import Tuple, Optional, List, Dict
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans, Birch
import warnings

# ============================================================
# PART 3: HIERARCHICAL CLUSTERING
# ============================================================

def compute_linkage_from_centroids(features: np.ndarray, labels: np.ndarray, linkage_method: str = 'ward') -> Optional[np.ndarray]:
    """
    Compute linkage matrix from cluster centroids instead of all samples.
    This is memory-efficient and shows relationships between final clusters.
    
    Args:
        features: (n_samples, n_features) array
        labels: cluster labels for each sample
        linkage_method: linkage method to use
        
    Returns:
        linkage_matrix: linkage matrix computed from centroids (or None if error)
    """
    try:
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        print(f"📊 Computing dendrogram from {n_clusters} cluster centroids...")
        
        # Compute cluster centroids
        centroids = np.array([features[labels == i].mean(axis=0) for i in unique_labels])
        
        # Compute linkage on centroids
        linkage_matrix = linkage(centroids, method=linkage_method)
        print(f"✓ Centroid-based linkage matrix computed successfully")
        
        return linkage_matrix
    except Exception as e:
        print(f"⚠ Error computing centroid linkage: {type(e).__name__}")
        return None

def run_hierarchical_clustering(
    features: np.ndarray,
    n_clusters: int = 10,
    linkage_method: str = 'ward',
    distance_threshold: float = None,
    max_samples_for_linkage: int = 10000,
    memory_efficient: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Run agglomerative hierarchical clustering with memory-efficient handling for large datasets.
    
    This function can handle datasets with hundreds of thousands of samples by:
    1. Computing the linkage matrix (dendrogram) on a subsample if dataset is large
    2. Using sklearn's memory-efficient AgglomerativeClustering on ALL samples
    3. Optionally falling back to MiniBatchKMeans for extremely large datasets
    
    Args:
        features: (n_samples, n_features) - should be scaled
        n_clusters: number of clusters (ignored if distance_threshold is set)
        linkage_method: 'ward', 'complete', 'average', 'single'
        distance_threshold: if set, determines clusters by cutting dendrogram
        max_samples_for_linkage: max samples for computing linkage matrix (dendrogram)
                                 Set to None to skip linkage computation entirely
        memory_efficient: if True and clustering fails due to memory, falls back to MiniBatchKMeans
        random_state: random seed for sampling
        
    Returns:
        labels: cluster assignments for ALL input samples (shape: n_samples,)
        linkage_matrix: for dendrogram plotting (computed on subsample if n_samples > max_samples_for_linkage)
                       Returns None if max_samples_for_linkage is None or computation is skipped
                       
    Note:
        - ALL samples are clustered, regardless of dataset size
        - Only the dendrogram visualization uses a subsample for large datasets
        - For datasets > 100k samples, AgglomerativeClustering may still be slow
    """
    n_samples, n_features = features.shape
    print(f"\n{'='*60}")
    print(f"Hierarchical Clustering: {n_samples:,} samples × {n_features} features")
    print(f"{'='*60}")
    
    # Compute linkage matrix for dendrogram (on subsample if needed)
    linkage_matrix = None
    if max_samples_for_linkage is not None and n_samples > max_samples_for_linkage:
        print(f"📊 Computing dendrogram on {max_samples_for_linkage:,} sampled points...")
        rng = np.random.RandomState(random_state)
        sample_idx = rng.choice(n_samples, size=max_samples_for_linkage, replace=False)
        features_sample = features[sample_idx]
        
        # Try progressively smaller subsamples if memory error occurs
        for attempt_size in [max_samples_for_linkage, 5000, 2000, 1000]:
            if attempt_size > len(features_sample):
                continue
            try:
                if attempt_size < max_samples_for_linkage:
                    print(f"  Retrying with {attempt_size:,} samples...")
                    sample_idx_smaller = rng.choice(len(features_sample), size=attempt_size, replace=False)
                    features_subsample = features_sample[sample_idx_smaller]
                else:
                    features_subsample = features_sample
                    
                linkage_matrix = linkage(features_subsample, method=linkage_method)
                print(f"✓ Linkage matrix computed successfully on {attempt_size:,} samples")
                break
            except (MemoryError, Exception) as e:
                if attempt_size == 1000:
                    print(f"⚠ Could not compute linkage even with 1000 samples - dendrogram unavailable")
                    print(f"  Error: {type(e).__name__}")
                    linkage_matrix = None
                continue
                
    elif max_samples_for_linkage is not None:
        print(f"📊 Computing dendrogram on full dataset...")
        try:
            linkage_matrix = linkage(features, method=linkage_method)
            print(f"✓ Linkage matrix computed successfully")
        except (MemoryError, Exception) as e:
            print(f"⚠ Error computing linkage - dendrogram will not be available")
            print(f"  Error: {type(e).__name__}: {str(e)[:100]}")
            linkage_matrix = None
    else:
        print("⊗ Skipping linkage matrix computation (dendrogram disabled)")
    
    # Run clustering on FULL dataset
    print(f"\n🔄 Clustering ALL {n_samples:,} samples...")
    
    try:
        # Try AgglomerativeClustering first (true hierarchical clustering)
        if distance_threshold is not None:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                linkage=linkage_method,
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage_method,
            )
        
        labels = clustering.fit_predict(features)
        print(f"✓ Hierarchical clustering complete!")
        
    except MemoryError as e:
        if not memory_efficient:
            raise
        
        # Fallback to MiniBatchKMeans for extremely large datasets
        warnings.warn(
            f"AgglomerativeClustering failed due to memory constraints. "
            f"Falling back to Birch clustering.",
            UserWarning
        )
        print(f"\n⚠ Memory error with hierarchical clustering")
        
        print(f"🔄 Falling back to MiniBatchKMeans...")
        #MiniBatchKMeans with early stopping based on convergence
        # kmeans = MiniBatchKMeans(
        #     n_clusters=n_clusters,
        #     random_state=random_state,
        #     batch_size=10000,
        #     max_iter=100,
        #     verbose=1,
        #     max_no_improvement=20,
        # )
        # labels = kmeans.fit_predict(features)
        # print(f"✓ MiniBatchKMeans clustering complete!")

        # Birch builds a tree and then uses AgglomerativeClustering internally
        print(f"🔄 Clustering with Birch...")
        birch = Birch(
            n_clusters=n_clusters,
            threshold=15, # You may need to tune this based on your scaled data - was 20 /15. my current results have 15 
            branching_factor=50
        )
        labels = birch.fit_predict(features)
        print(f"✓ Birch clustering complete!")

        
        linkage_matrix = None  # No dendrogram for KMeans
    
    n_clusters_found = len(np.unique(labels))
    print(f"\n📈 Results: {n_clusters_found} clusters found")
    for i in range(n_clusters_found):
        count = np.sum(labels == i)
        pct = 100 * count / n_samples
        print(f"   Cluster {i}: {count:,} samples ({pct:.1f}%)")
    
    # If linkage matrix wasn't computed, try computing from centroids
    if linkage_matrix is None and n_clusters_found > 1:
        print(f"\n💡 Attempting to compute dendrogram from cluster centroids...")
        linkage_matrix = compute_linkage_from_centroids(features, labels, linkage_method)
    
    print(f"{'='*60}\n")
    
    return labels, linkage_matrix

# give clusters names based on the features


def assign_cluster_names(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str]
) -> Dict[int, str]:
    """
    Programmatically assign interpretable names to clusters based on robust feature deviations.
    """
    cluster_names = {}
    n_clusters = len(np.unique(labels))
    
    # Calculate dataset-wide medians and IQRs to avoid the "96% stationary" skew
    q75, q25 = np.percentile(features, [75, 25], axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1e-6  # Prevent division by zero
    dataset_median = np.median(features, axis=0)
    
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_features = features[mask]
        
        # Get the median of THIS cluster to avoid internal cluster outliers
        cluster_median = np.median(cluster_features, axis=0)
        
        # Calculate how many IQRs away from the dataset baseline this cluster is
        # This gives us a "Robust Z-score" for every feature
        deviations = (cluster_median - dataset_median) / iqr
        dev_dict = {name: dev for name, dev in zip(feature_names, deviations)}
        
        characteristics = []
        
        # 1. Is it the stationary cluster?
        # Check if all speed and acceleration deviations are close to baseline
        speed_accel_devs = [dev_dict[k] for k in feature_names if 'speed' in k or 'acc' in k]
        if speed_accel_devs and np.max(speed_accel_devs) < 1.0:
            cluster_names[cluster_id] = f"C{cluster_id}_Stationary"
            continue
            
        # 2. Check for explosive acceleration
        accel_devs = [dev_dict[k] for k in feature_names if 'acc' in k]
        if accel_devs and np.max(accel_devs) > 3.0:
            characteristics.append("High_Acceleration")
            
        # 3. Check movement directions (vy = wheel running, vz = vertical lift)
        vy_devs = [dev_dict[k] for k in feature_names if 'vy' in k]
        vz_devs = [dev_dict[k] for k in feature_names if 'vz' in k]
        
        avg_vy = np.mean(vy_devs) if vy_devs else 0
        avg_vz = np.mean(vz_devs) if vz_devs else 0
        
        if avg_vz > 2.0 and avg_vz > avg_vy:
            characteristics.append("Vertical_Movement")
        elif avg_vy > 2.0:
            characteristics.append("Running")
            
        # 4. Check Asymmetry (Left vs Right paw dominance)
        pawL_speed = dev_dict.get('pawL_speed', 0)
        pawR_speed = dev_dict.get('pawR_speed', 0)
        
        if pawL_speed > pawR_speed + 1.5:
            characteristics.append("Left_Paw_Dominant")
        elif pawR_speed > pawL_speed + 1.5:
            characteristics.append("Right_Paw_Dominant")
        elif pawL_speed > 1.5 and pawR_speed > 1.5:
            if "Running" not in characteristics and "Vertical_Movement" not in characteristics:
                characteristics.append("Bilateral_Active")
        
        # Build the final name string
        if characteristics:
            name = f"C{cluster_id}_{'+'.join(characteristics)}"
        else:
            name = f"C{cluster_id}_Unclassified_Active"
            
        cluster_names[cluster_id] = name
        
    return cluster_names












# import numpy as np
# import warnings
# from typing import Tuple, Optional
# from scipy.cluster.hierarchy import linkage, fcluster


# # ============================================================
# # PART 3: HIERARCHICAL CLUSTERING
# # ============================================================

# def _assign_noise_to_nearest(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
#     """Assign HDBSCAN noise points (label == -1) to nearest cluster centroid."""
#     labels = labels.copy()
#     valid_labels = np.unique(labels[labels >= 0])
#     centroids = np.stack([features[labels == k].mean(axis=0) for k in valid_labels])
#     noise_idx = np.where(labels == -1)[0]
#     for start in range(0, len(noise_idx), 10_000):
#         end = min(start + 10_000, len(noise_idx))
#         dists = np.linalg.norm(
#             features[noise_idx[start:end], None, :] - centroids[None, :, :], axis=2
#         )
#         labels[noise_idx[start:end]] = valid_labels[dists.argmin(axis=1)]
#     return labels


# def _mst_to_linkage(edges: np.ndarray, n: int) -> np.ndarray:
#     """Convert MST edge list [(i, j, dist), ...] to scipy linkage matrix."""
#     edges_sorted = edges[np.argsort(edges[:, 2])]
#     parent = np.arange(2 * n - 1)
#     size   = np.ones(2 * n - 1)

#     def find(x):
#         while parent[x] != x:
#             parent[x] = parent[parent[x]]
#             x = parent[x]
#         return x

#     Z = np.zeros((n - 1, 4))
#     next_id = n
#     for idx, (i, j, dist) in enumerate(edges_sorted):
#         ci, cj = find(int(i)), find(int(j))
#         if ci == cj:
#             continue
#         Z[idx] = [ci, cj, dist, size[ci] + size[cj]]
#         parent[ci] = parent[cj] = next_id
#         size[next_id] = size[ci] + size[cj]
#         next_id += 1
#     return Z


# def compute_linkage_from_centroids(
#     features: np.ndarray,
#     labels: np.ndarray,
#     linkage_method: str = 'ward',
# ) -> Optional[np.ndarray]:
#     """Compute linkage matrix from cluster centroids for dendrogram plotting."""
#     try:
#         unique_labels = np.unique(labels)
#         print(f"📊 Computing dendrogram from {len(unique_labels)} cluster centroids...")
#         centroids = np.array([features[labels == i].mean(axis=0) for i in unique_labels])
#         Z = linkage(centroids, method=linkage_method)
#         print(f"✓ Centroid-based linkage matrix computed successfully")
#         return Z
#     except Exception as e:
#         print(f"⚠ Error computing centroid linkage: {type(e).__name__}")
#         return None


# def run_hierarchical_clustering(
#     features: np.ndarray,
#     n_clusters: int = 10,
#     linkage_method: str = 'ward',       # used for centroid dendrogram only
#     distance_threshold: float = None,   # kept for API compatibility, not used
#     max_samples_for_linkage: int = 10000,  # kept for API compatibility, not used
#     memory_efficient: bool = True,      # kept for API compatibility, not used
#     random_state: int = 42,             # kept for API compatibility, not used
#     # HDBSCAN tuning
#     min_cluster_size: int = 500,
#     min_samples: int = 50,
# ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
#     """
#     Hierarchical behavioral clustering via HDBSCAN + condensed-tree flat cut.

#     Replaces the old AgglomerativeClustering/MiniBatchKMeans approach which
#     either ran out of memory or silently fell back to non-hierarchical KMeans.

#     Why HDBSCAN:
#       - Runs on ALL 400k samples (no subsampling, no approximation)
#       - Density-aware: correctly isolates rare movement states even when the
#         dataset is dominated by still frames. Ward linkage would just keep
#         bisecting the large still blob and miss small movement clusters.
#       - Full condensed tree is built once; we cut it at exactly n_clusters
#         so you get true hierarchical structure with controlled k.

#     Args:
#         features:          (n_samples, n_features) scaled feature array
#         n_clusters:        exact number of clusters to extract
#         linkage_method:    linkage used for the centroid dendrogram plot
#         min_cluster_size:  min frames to form a cluster (start at 500,
#                            increase if you see too many tiny clusters)
#         min_samples:       noise sensitivity (increase if too many -1 labels)
#         (remaining args kept for drop-in API compatibility, not used internally)

#     Returns:
#         labels:         (n_samples,) int32 cluster assignments, 0-indexed, no noise
#         linkage_matrix: centroid linkage matrix for dendrogram plotting
#     """
#     n_samples, n_features = features.shape
#     print(f"\n{'='*60}")
#     print(f"Hierarchical Clustering: {n_samples:,} samples × {n_features} features")
#     print(f"Target clusters: {n_clusters}  |  min_cluster_size: {min_cluster_size}")
#     print(f"{'='*60}")

#     features = features.astype(np.float32)

#     # ── Step 1: Run HDBSCAN (GPU via cuML if available, else CPU) ─────────
#     use_cuml = False
#     try:
#         import cudf
#         from cuml.cluster import HDBSCAN as cuHDBSCAN
#         print("🚀 Using cuML GPU-accelerated HDBSCAN")
#         clusterer = cuHDBSCAN(
#             min_cluster_size=min_cluster_size,
#             min_samples=min_samples,
#             cluster_selection_method='eom',
#             gen_min_span_tree=True,
#             prediction_data=True,
#         )
#         clusterer.fit(cudf.DataFrame(features))
#         raw_labels = clusterer.labels_.to_numpy()
#         use_cuml = True
#     except ImportError:
#         print("🖥  cuML not found — using CPU hdbscan")
#         print("   To enable GPU: conda install -c rapidsai cuml")
#         import hdbscan as hdbscan_lib
#         clusterer = hdbscan_lib.HDBSCAN(
#             min_cluster_size=min_cluster_size,
#             min_samples=min_samples,
#             cluster_selection_method='eom',
#             core_dist_n_jobs=-1,
#             gen_min_span_tree=True,
#             prediction_data=True,
#         )
#         raw_labels = clusterer.fit_predict(features)

#     n_raw   = len(np.unique(raw_labels[raw_labels >= 0]))
#     n_noise = int(np.sum(raw_labels == -1))
#     print(f"✓ HDBSCAN complete: {n_raw} raw clusters, {n_noise:,} noise points")

#     # ── Step 2: Cut condensed tree at exactly n_clusters ──────────────────
#     print(f"\n🌲 Cutting condensed tree → {n_clusters} clusters...")
#     try:
#         mst   = clusterer.minimum_spanning_tree_
#         edges = mst.to_pandas().values if use_cuml else np.array(mst)
#         Z     = _mst_to_linkage(edges, n_samples)
#         labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1  # 0-indexed
#         print(f"✓ Tree cut successful")
#     except Exception as e:
#         warnings.warn(f"Tree cut failed ({e}), falling back to raw HDBSCAN labels.")
#         labels = raw_labels

#     # ── Step 3: Assign any remaining noise points to nearest centroid ──────
#     n_noise_remaining = int(np.sum(labels == -1))
#     if n_noise_remaining > 0:
#         print(f"  Assigning {n_noise_remaining:,} noise frames to nearest centroid...")
#         labels = _assign_noise_to_nearest(features, labels)

#     # Remap to contiguous 0-indexed labels
#     unique = np.unique(labels)
#     remap  = {old: new for new, old in enumerate(unique)}
#     labels = np.array([remap[l] for l in labels], dtype=np.int32)

#     # ── Step 4: Centroid linkage for dendrogram ────────────────────────────
#     linkage_matrix = compute_linkage_from_centroids(features, labels, linkage_method)

#     # ── Summary ───────────────────────────────────────────────────────────
#     n_found = len(np.unique(labels))
#     print(f"\n📈 Results: {n_found} clusters across {n_samples:,} frames")
#     for i in range(n_found):
#         count = int(np.sum(labels == i))
#         pct   = 100 * count / n_samples
#         bar   = '█' * int(pct / 2)
#         print(f"   Cluster {i}: {count:7,} samples ({pct:5.1f}%)  {bar}")
#     print(f"{'='*60}\n")

#     return labels, linkage_matrix