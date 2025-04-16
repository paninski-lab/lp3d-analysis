import os
import re
import pandas as pd
import numpy as np
import pickle 
import traceback

from typing import Optional, Union, Callable

from lp3d_analysis import io
from omegaconf import DictConfig
from typing import List, Literal, Tuple, Dict, Any 
from pathlib import Path

from lightning_pose.utils import io as io_utils
# from lightning_pose.utils.cropzoom import generate_cropped_csv_file # the old generated csv file
from lightning_pose.utils.scripts import (
    compute_metrics,
)
from lightning_pose.utils import io as io_utils

from eks.singlecam_smoother import ensemble_kalman_smoother_singlecam, fit_eks_singlecam
from eks.multicam_smoother import ensemble_kalman_smoother_multicam
from eks.utils import convert_lp_dlc, format_data, make_dlc_pandas_index
from eks.core import jax_ensemble
from eks.marker_array import MarkerArray, input_dfs_to_markerArray

from lp3d_analysis.utils import (
    add_variance_columns,
    fill_ensemble_results,
    prepare_uncropped_csv_files,
    generate_cropped_csv_file,
)

from lp3d_analysis.io import (
    # collect_files_by_token,
    process_predictions,
    setup_ensemble_dirs,
    # get_original_structure,
    # collect_csv_files_by_seed,
    # group_directories_by_sequence
)

'''
This is loading the pca and FA objects - we will probably not use it like that later 
'''


# # loading pca objects and fa objects if necessart
# # pca_model_path = "/teamspace/studios/this_studio/ZZ_pca_objects/pca_object_inD_mirror-mouse-separate.pkl"
pca_model_path = "/teamspace/studios/this_studio/ZZ_pca_objects/pca_object_inD_fly.pkl"
# pca_model_path = "/teamspace/studios/this_studio/ZZ_pca_objects/pca_object_inD_chickadee-crop_6pcs_new.pkl"

# # fa_model_path = "/teamspace/studios/this_studio/ZZ_pca_objects/pca_object_inD_mirror-mouse-separate.pkl"
fa_model_path = "/teamspace/studios/this_studio/ZZ_pca_objects/fa_object_inD_fly.pkl"
# fa_model_path = "/teamspace/studios/this_studio/ZZ_pca_objects/pca_object_inD_chickadee-crop_6pcs_new.pkl"

class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler to handle special model classes"""
    def find_class(self, module, name):
        if name == "NaNPCA":
            from lightning_pose.utils.pca import NaNPCA
            return NaNPCA
        elif name == "EnhancedFactorAnalysis":
            from lp3d_analysis.pca_global import EnhancedFactorAnalysis
            return EnhancedFactorAnalysis
        return super().find_class(module, name)


def load_models(pca_path=None, fa_path=None):
    """
    Load PCA and FA models from pickle files.
    
    Args:
        pca_path: Path to PCA model pickle file (optional)
        fa_path: Path to FA model pickle file (optional)
    
    Returns:
        Tuple of (pca_object, fa_object) - either model can be None if loading fails
    """
    pca_object = None
    fa_object = None
    
    # Try to load PCA model
    if pca_path:
        try:
            with open(pca_path, "rb") as f:
                pca_object = CustomUnpickler(f).load()
            print(f"PCA model loaded successfully from {pca_path}")
        except (AttributeError, FileNotFoundError) as e:
            print(f"Error loading PCA model from {pca_path}: {e}")
    
    # Try to load FA model
    if fa_path:
        try:
            with open(fa_path, "rb") as f:
                fa_object = CustomUnpickler(f).load()
            print(f"FA model loaded successfully from {fa_path}")
        except (AttributeError, FileNotFoundError) as e:
            print(f"Error loading FA model from {fa_path}: {e}")
    
    return pca_object, fa_object

# Load PCA and FA models
if 'pca_model_path' in locals() and pca_model_path:
    pca_object, _ = load_models(pca_path=pca_model_path)
if 'fa_model_path' in locals() and fa_model_path:
    _, fa_object = load_models(fa_path=fa_model_path)

def _load_latent_models(pca_object=None, fa_object=None):
    """Helper function to load PCA and FA models from globals if not provided"""
    pca_object = pca_object or globals().get('pca_object')
    fa_object = fa_object or globals().get('fa_object')
    return pca_object, fa_object



def _get_bbox_path_fn(p: Path, results_dir: Path, data_dir: Path) -> Path:
    """Given some preds file `p` and `results_dir`, it will return p's
    corresponding bbox path in the data directory."""
    # Gets the current depth in the model directory.
    n_levels_up = len(p.parent.parts) - len(results_dir.parts)
    # Hack: correct the logic for when transforming results of eks_multiview back to cropped space.
    if "eks_multiview" in p.parts:
        n_levels_up -= 1
    # Get `p` relative to model_dir
    model_dir = p.parents[n_levels_up]
    relative_p = Path(p).relative_to(model_dir)
    p_in_data_dir = data_dir / relative_p
    bbox_path = p_in_data_dir.with_stem(p.stem + "_bbox")
    return bbox_path


def process_ensemble_frames(
    all_pred_files: List[str],
    keypoint_names: List[str],
    mode: str = "ensemble_mean"
) -> pd.DataFrame:
    """Process ensemble frames using the specified mode, works for ensemble_mean and ensemble_median"""
    # For single view modes
    input_dfs_list, keypoint_names = format_data(
        input_source=all_pred_files,
        camera_names=None,
    )
    
    # Process predictions to get column structure
    column_structure, _, _, _, _ = process_predictions(all_pred_files[0])
    
    # Stack arrays for processing
    stacked_arrays = []
    stacked_dfs = []
    df_index = None
    
    for markers_curr_fmt in input_dfs_list:
        if df_index is None:
            df_index = markers_curr_fmt.index
        array_data = markers_curr_fmt.to_numpy()
        numeric_cols = markers_curr_fmt.select_dtypes(include=[np.number])
        stacked_arrays.append(array_data)
        stacked_dfs.append(numeric_cols)
    
    stacked_arrays = np.stack(stacked_arrays, axis=-1)
    
    if mode in ['ensemble_mean', 'ensemble_median']:
        # Process with jax_ensemble
        stacked_arrays_reshaped = stacked_arrays.transpose(2, 0, 1)  # (5, 601, 90)
        stacked_arrays_reshaped = input_dfs_to_markerArray([stacked_dfs], keypoint_names, camera_names=[""])
        avg_mode = 'mean' if mode == 'ensemble_mean' else 'median'
        
        ensemble_marker_array = jax_ensemble(
            stacked_arrays_reshaped, 
            avg_mode=avg_mode, 
            var_mode='confidence_weighted_var',
        )

        ensemble_preds = ensemble_marker_array.slice_fields("x", "y").get_array(squeeze=True)
        ensemble_vars = ensemble_marker_array.slice_fields("var_x", "var_y").get_array(squeeze=True)
        ensemble_likes = ensemble_marker_array.slice_fields("likelihood").get_array(squeeze=True)
        # Add variance columns to the structure
        column_structure = add_variance_columns(column_structure)
        results_df = pd.DataFrame(index=df_index, columns=column_structure)
        
        # Fill in values for each keypoint
        fill_ensemble_results(
            results_df, 
            ensemble_preds, 
            ensemble_vars, 
            ensemble_likes, 
            keypoint_names, 
            column_structure
        )
        
    return results_df



def process_multiple_video_single_view(
    view: str,
    seed_dirs: List[str],
    inference_dir: str,
    output_dir: str,
    mode: str
) -> None:
    """Process multiple video files for a single view"""
    
    # Get all CSV files from the first seed directory
    first_seed_dir = seed_dirs[0]
    base_files = os.listdir(os.path.join(first_seed_dir, inference_dir))
    view_files = [f for f in base_files if view in f and f.endswith('.csv')] # all csv files with view name 
    # Process each file (representing a different video) separately
    for file_name in view_files:
        print(f"Processing file: {file_name}")
        
        # Collect this file from all seed directories
        pred_files = []
        for seed_dir in seed_dirs:
            pred_file = os.path.join(seed_dir, inference_dir, file_name)
            if os.path.exists(pred_file):
                pred_files.append(pred_file)
            
        # Process ensemble data
        input_dfs_list, keypoint_names = format_data(
            input_source=pred_files,
            camera_names=None,
        )
        
        if mode in ['ensemble_mean', 'ensemble_median']:
            results_df = process_ensemble_frames(pred_files, keypoint_names, mode=mode)
        elif mode == 'eks_singleview':
            results_df = run_eks_singleview(
                markers_list=input_dfs_list,
                keypoint_names=keypoint_names
            )
        else:
            print(f"Invalid mode: {mode}")
            continue
        
        # Save results using the same filename
        preds_file = os.path.join(output_dir, file_name)
        results_df.to_csv(preds_file)
        print(f"Saved ensemble {mode} predictions for {file_name} to {preds_file}")


def post_process_ensemble_videos(
    cfg_lp: DictConfig,
    results_dir: str,
    model_type: str,
    n_labels: int,
    seed_range: tuple[int, int],
    views: List[str], 
    mode: Literal['ensemble_mean', 'ensemble_median', 'eks_singleview', 'eks_multiview'],
    inference_dirs: List[str],
    overwrite: bool,
    n_latent: int = 3,
    pca_object = None,
    fa_object = None
) -> None:
    """Post-process ensemble videos"""
    
    # Load models if needed
    pca_object, fa_object = _load_latent_models(pca_object, fa_object)
    print(f"Using PCA object: {pca_object}")
    print(f"Using FA object: {fa_object}")

    # Prepare FA parameters for inflation if FA object is available
    inflate_vars_kwargs = {}
    if fa_object is not None:
        # Extract loading matrix and mean from the FA object
        print("Extracting FA parameters for variance inflation")
        try:
            loading_matrix = fa_object.components_.T  # Typically the loading matrix is the transpose of components_
            mean = fa_object.mean_
            
            inflate_vars_kwargs = {
                'loading_matrix': loading_matrix,
                # 'mean': mean,
                'mean': np.zeros_like(mean)  # we had an issue of centering twice 
            }
            print("Successfully extracted FA parameters for variance inflation")
        except AttributeError as e:
            print(f"Error extracting FA parameters: {e}. Using default inflation parameters.")

    # Setup directories
    base_dir = os.path.dirname(results_dir)
    ensemble_dir, seed_dirs, _ = setup_ensemble_dirs(
        base_dir, model_type, n_labels, seed_range, mode, ""
    )
    
    for inference_dir in inference_dirs:
        print(f"Post-processing ensemble predictions for {model_type} {n_labels} {seed_range[0]}-{seed_range[1]} for {inference_dir}")
        first_seed_dir = seed_dirs[0]
        inf_dir_in_first_seed = os.path.join(first_seed_dir, inference_dir)
        
        # Check if the inference_dir contains ANY subdirectories
        # entries: sort so that sessions are in the same order for each view.
        entries = list(sorted(filter(lambda f: not f.endswith("_uncropped.csv"), os.listdir(inf_dir_in_first_seed))))
        contains_subdirectory = any(os.path.isdir(os.path.join(inf_dir_in_first_seed, entry)) for entry in entries)
        
        if contains_subdirectory:
            print(f"Found subdirectories in {inf_dir_in_first_seed}. Skipping.")
            continue  # Skip if there are any subdirectories

        print(f"Directory contains only files. Processing {inference_dir}...")
        output_dir = os.path.join(ensemble_dir, mode, inference_dir)
        os.makedirs(output_dir, exist_ok=True)

        if mode == 'eks_multiview':
            # Get the files in the inference directory partitioned by view.
            files_by_view = io.collect_files_by_token(list(map(Path, entries)), views)
            # Get the files in the inference directory FOR JUST ONE VIEW.
            # There should be one file per session. These we call "first view sessions".
            first_view_sessions = list(map(str, files_by_view[views[0]]))
            for first_view_session in first_view_sessions:
                # Process multiview case for videos
                all_pred_files = []

                # Collect files for all views and seeds
                for view in views:
                    print(f"Processing view: {view}")
                    # Get the filename of the prediction file for this session-view pair.
                    session = first_view_session.replace(views[0], view)

                    for seed_dir in seed_dirs:
                        all_pred_files.append(os.path.join(seed_dir, inference_dir, session))

                # Process ensemble data

                # Not None when there's bbox files in the dataset, i.e. chickadee-crop.
                all_pred_files_uncropped = prepare_uncropped_csv_files(all_pred_files, lambda p: _get_bbox_path_fn(p, Path(results_dir), Path(cfg_lp.data.data_dir)))


                input_dfs_list, keypoint_names = format_data(
                    input_source=all_pred_files_uncropped or all_pred_files,
                    camera_names=views,
                )

                # Run multiview EKS
                results_dfs = run_eks_multiview(
                    markers_list=input_dfs_list,
                    keypoint_names=keypoint_names,
                    views=views,
                    quantile_keep_pca = 50,
                    inflate_vars_kwargs=inflate_vars_kwargs,
                    n_latent=n_latent,
                    pca_object=pca_object,
                )

                # Save results for each view
                for view in views:
                    session = first_view_session.replace(views[0], view)
                    result_df = results_dfs[view]
                    result_file = Path(output_dir) / session
                    # If cropped dataset, save to a _uncropped file, so that we can later undo the remapping.
                    if all_pred_files_uncropped is not None:
                        uncropped_result_file = result_file.with_stem(result_file.stem + "_uncropped")
                    else:
                        uncropped_result_file = None

                    result_df.to_csv(uncropped_result_file or result_file)

                    # Crop the multiview-eks output back to original cropped coordinate space.
                    if all_pred_files_uncropped is not None:
                        bbox_path = _get_bbox_path_fn(result_file, Path(results_dir), Path(cfg_lp.data.data_dir))
                        generate_cropped_csv_file(uncropped_result_file, bbox_path, result_file, img_height = 320, img_width = 320, mode="subtract")

                    print(f"Saved ensemble {mode} predictions for {view} view to {result_file}")
        else:
            # Process each view separately for other modes
            for view in views:
                process_multiple_video_single_view(
                    view=view,
                    seed_dirs=seed_dirs,
                    inference_dir=inference_dir,
                    output_dir=output_dir,
                    mode=mode
                )

def extract_labeled_frame_predictions(
    cfg_lp: DictConfig,
    results_dir: str,
    model_type: str,
    n_labels: int,
    seed_range: tuple[int, int],
    views: List[str],
    mode: str,
    inference_dir: str = "videos_new",
    overwrite: bool = False
) -> None:
    """
    Extract predictions specifically for labeled frames from a method's output.
    
    Args:
        cfg_lp: Lightning Pose configuration
        results_dir: Results directory path
        model_type: Model type (e.g., 'supervised')
        n_labels: Number of labels used for training
        seed_range: Range of seeds used (e.g., (0, 4))
        views: List of camera view names
        mode: Processing method ('ensemble_mean', 'ensemble_median', 'eks_singleview', 'eks_multiview')
        inference_dir: Name of the directory containing video predictions (default: "videos_new")
        overwrite: Whether to overwrite existing files
    """

    # Setup paths
    base_dir = os.path.dirname(results_dir)
    ensemble_dir, seed_dirs, _ = setup_ensemble_dirs(
        base_dir, model_type, n_labels, seed_range, mode, ""
    )
    
    method_dir = os.path.join(ensemble_dir, mode)
    path_videos_new = os.path.join(method_dir, inference_dir)
    
    # Create output directory for labeled frame predictions
    output_dir = os.path.join(method_dir, "videos-for-each-labeled-frame")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if predictions already exist and should not be overwritten
    if not overwrite:
        for view in views:
            output_file = os.path.join(output_dir, f"predictions_{view}_new.csv")
            if os.path.exists(output_file):
                print(f"Predictions file {output_file} already exists. Skipping (overwrite=False).")
                return
    
    print(f"Processing predictions from {path_videos_new}")
    print(f"Saving results to {output_dir}")
    
    # Process each view
    for view in views:
        print(f"\nProcessing {view}...")
        
        # Find the CollectedData file for this view
        collected_data_path = os.path.join(cfg_lp.data.data_dir, f"CollectedData_{view}_new.csv")
        if not os.path.exists(collected_data_path):
            print(f"Warning: CollectedData file not found for {view}")
            continue
        
        try:
            # Load the original CollectedData file to get the index structure
            collected_df = pd.read_csv(collected_data_path, header=[0, 1, 2], index_col=0)
            collected_df = io_utils.fix_empty_first_row(collected_df)
            print(f"Loaded CollectedData file with shape: {collected_df.shape}")
            
            # Find all valid image indices
            valid_indices = [idx for idx in collected_df.index 
                            if idx not in ['bodyparts', 'coords'] 
                            and isinstance(idx, str) 
                            and '/' in idx]
            
            print(f"Found {len(valid_indices)} valid indices in CollectedData")
            
            # Group indices by sequence
            sequences = {}
            for idx in valid_indices:
                parts = idx.split('/')
                if len(parts) >= 2:
                    sequence = parts[1]  # e.g., "180623_000_bot"
                    if sequence not in sequences:
                        sequences[sequence] = []
                    sequences[sequence].append(idx)
            
            print(f"Found {len(sequences)} unique sequences")
            
            # Find prediction files for this view
            prediction_files = []
            for filename in os.listdir(path_videos_new):
                if os.path.isfile(os.path.join(path_videos_new, filename)):
                    # Match view name as a standalone token
                    pattern = rf"(?<![A-Z]){re.escape(view)}(?![A-Z])"
                    if re.search(pattern, filename) and 'pixel_error' not in filename:
                        prediction_files.append(os.path.join(path_videos_new, filename))
            
            print(f"Found {len(prediction_files)} prediction files for {view}")
            
            # Create a DataFrame for combined results
            combined_df = None
            
            # Process each sequence
            for sequence, indices in sequences.items():
                print(f"\nProcessing sequence: {sequence}")
                
                # Find prediction files for this sequence
                sequence_pred_files = [f for f in prediction_files if sequence in os.path.basename(f)]
                print(f"Found {len(sequence_pred_files)} prediction files for sequence {sequence}")
                
                if not sequence_pred_files:
                    print(f"No prediction files found for sequence {sequence}")
                    continue
                
                # Process each prediction file
                for pred_file in sequence_pred_files:
                    print(f"Processing file: {os.path.basename(pred_file)}")
                    
                    try:
                        # Load the prediction file
                        pred_df = pd.read_csv(pred_file, header=[0, 1, 2], index_col=0)
                        pred_df = io_utils.fix_empty_first_row(pred_df)
                        print(f"Loaded prediction file with shape: {pred_df.shape}")
                        
                        # Initialize combined_df if needed
                        if combined_df is None:
                            combined_df = pd.DataFrame(index=collected_df.index, columns=pred_df.columns)
                            print(f"Initialized combined DataFrame with shape: {combined_df.shape}")
                        
                        # Extract frame numbers from image paths
                        img_to_frame_num = {}
                        for idx in indices:
                            parts = idx.split('/')
                            if len(parts) >= 3:
                                frame_file = parts[2]
                                frame_match = re.search(r'img(\d+)\.png', frame_file)
                                if frame_match:
                                    frame_num = int(frame_match.group(1))
                                    img_to_frame_num[idx] = frame_num
                        
                        # Check prediction file index structure
                        if pred_df.index.dtype == 'int64' or all(isinstance(x, int) for x in pred_df.index if isinstance(x, (int, float))):
                            # Numeric indices (likely frame numbers)
                            print("Prediction file has numeric indices")
                            
                            # Check if these are actual frame numbers
                            sample_frames = list(img_to_frame_num.values())[:5] if img_to_frame_num else []
                            frame_exists = [frame in pred_df.index for frame in sample_frames]
                            
                            if any(frame_exists):
                                # Direct frame number matching
                                match_count = 0
                                for img_path, frame_num in img_to_frame_num.items():
                                    if frame_num in pred_df.index:
                                        combined_df.loc[img_path] = pred_df.loc[frame_num]
                                        match_count += 1
                                
                                print(f"Matched {match_count} frames by direct frame number")
                            else:
                                # Try position-based matching
                                print("Indices are not direct frame numbers, trying sequential mapping")
                                
                                sorted_pred_indices = sorted(pred_df.index)
                                sorted_img_paths = sorted(img_to_frame_num.keys(), key=lambda x: img_to_frame_num[x])
                                
                                # Check if lengths match
                                if len(sorted_pred_indices) == len(sorted_img_paths):
                                    print("Equal number of frames, using direct position mapping")
                                    
                                    for i in range(len(sorted_img_paths)):
                                        img_path = sorted_img_paths[i]
                                        pred_idx = sorted_pred_indices[i]
                                        combined_df.loc[img_path] = pred_df.loc[pred_idx]
                                    
                                    print(f"Mapped {len(sorted_img_paths)} frames by position")
                                else:
                                    print(f"Warning: Number of frames doesn't match. Prediction: {len(sorted_pred_indices)}, CollectedData: {len(sorted_img_paths)}")
                                    
                                    min_length = min(len(sorted_img_paths), len(sorted_pred_indices))
                                    for i in range(min_length):
                                        img_path = sorted_img_paths[i]
                                        pred_idx = sorted_pred_indices[i]
                                        combined_df.loc[img_path] = pred_df.loc[pred_idx]
                                    
                                    print(f"Mapped {min_length} frames by position")
                        else:
                            # String indices
                            print("Prediction file has string indices")
                            
                            # Check if indices contain sequence names
                            seq_indices = [idx for idx in pred_df.index if isinstance(idx, str) and sequence in idx]
                            
                            if seq_indices:
                                print("Indices contain sequence names")
                                
                                # Match by frame number in path
                                match_count = 0
                                for img_path, frame_num in img_to_frame_num.items():
                                    for pred_idx in seq_indices:
                                        pred_frame_match = re.search(r'img(\d+)\.png', pred_idx)
                                        if pred_frame_match and int(pred_frame_match.group(1)) == frame_num:
                                            combined_df.loc[img_path] = pred_df.loc[pred_idx]
                                            match_count += 1
                                            break
                                
                                print(f"Matched {match_count} frames by frame number in path")
                            else:
                                print("Warning: Could not determine how to map indices for this file")
                    
                    except Exception as e:
                        print(f"Error processing prediction file {pred_file}: {e}")
                        print(traceback.format_exc())
            
            # Save and evaluate combined results
            if combined_df is not None:
                # Add "set" column for labeled data
                combined_df.loc[:,("set", "", "")] = "train"
                
                # Count non-empty rows
                non_empty_count = combined_df.dropna(how='all').shape[0]
                print(f"\nSuccessfully filled {non_empty_count} out of {combined_df.shape[0]} rows")
                
                # Save predictions
                output_file = os.path.join(output_dir, f"predictions_{view}_new.csv")
                combined_df.to_csv(output_file)
                print(f"Saved predictions to {output_file}")
                
                # Compute metrics
                cfg_lp_view = cfg_lp.copy()
                cfg_lp_view.data.csv_file = [f'CollectedData_{view}_new.csv']
                cfg_lp_view.data.view_names = [view]
                
                try:
                    compute_metrics(cfg=cfg_lp_view, preds_file=[output_file], data_module=None)
                    print(f"Successfully computed metrics for {view}")
                except Exception as e:
                    print(f"Error computing metrics for {view}: {str(e)}")
                    print(traceback.format_exc())
            else:
                print(f"Warning: No data was combined for {view}")
        
        except Exception as e:
            print(f"Error processing view {view}: {e}")
            print(traceback.format_exc())
    
    print("\nAll views processed!")


def run_eks_singleview(
    markers_list: List[pd.DataFrame],
    keypoint_names : List[str], # can't I take it from the configrations?
    blocks : list = [], # will need to take care of that 
    avg_mode: str = 'median',
    var_mode : str = 'confidence_weighted_var',
  
) -> pd.DataFrame:
    # tuple[np.ndarray, pd.DataFrame]:
    """
    Process single view data using Ensemble Kalman Smoother.
    Args:
        markers_list: List of DataFrames containing predictions from different ensemble members
        keypoint_names: List of keypoint names to process
        blocks: List of keypoint blocks for correlated noise
        avg_mode: Mode for averaging across ensemble
        var_mode: Mode for computing ensemble variance
    Returns:
        pd.DataFrame: A DataFrame with the full smoothed data, including detailed
        statistics for all keypoints and their dimensions.
    """

    print(f'Input data loaded for keypoints: {keypoint_names}')
    print(f'Number of ensemble members: {len(markers_list)}')

    marker_array = input_dfs_to_markerArray([markers_list], keypoint_names, camera_names=[""])
    # Run the smoother with the simplified data
    results_df, smooth_params_final = ensemble_kalman_smoother_singlecam(
        marker_array= marker_array, # it was markers_list
        keypoint_names=keypoint_names,
        smooth_param= None, # estimating a smoothing param and it is computing the negative log likelihood of the sequence under the smoothing parametr --> this will be 10 for now 
        s_frames=None, # the frames using to actually run the optimization --> None for now but if we have a long video will probably use 10,000 frames 
        blocks=blocks, 
        avg_mode=avg_mode,
        var_mode=var_mode,

    )
    
    return results_df


def run_eks_multiview(
    markers_list: List[List[pd.DataFrame]],
    keypoint_names: List[str],
    views: List[str],
    blocks: list = [],
    smooth_param: Optional[Union[float, list]] = None,
    s_frames: Optional[list] = None,
    quantile_keep_pca: float = 50,
    avg_mode: str = 'median',
    var_mode: str = 'confidence_weighted_var',
    # inflate_vars_likelihood_thresh = None,
    # inflate_vars_v_quantile_thresh = None,
    inflate_vars_kwargs: dict = {},
    verbose: bool = False,
    n_latent: int =3,
    pca_object = None,
) -> dict[str, pd.DataFrame]:

    """
    Process multi-view data using Ensemble Kalman Smoother. This function handles multiple camera views
    and ensemble predictions for each keypoint.

    Args:
        markers_list: Nested list of DataFrames containing predictions from different ensemble members
                    for each camera view. Structure: [camera_view][ensemble_member]
        keypoint_names: List of keypoint names to process (e.g., ['nose', 'left_ear', 'right_ear'])
        views: List of camera view names (e.g., ['cam1', 'cam2', 'cam3'])
        blocks: List of keypoint blocks for correlated noise processing. Default is empty list.
        smooth_param: Smoothing parameter(s) for the Kalman filter. Can be a single float or list.
                    Default is None, which uses internal default value of 1000.
        s_frames: List of frames to use for optimization. Default is None (uses all frames).
        quantile_keep_pca: Quantile threshold for PCA component retention. Default is 95.
        avg_mode: Method for averaging across ensemble members. Default is 'median'.
        var_mode: Method for computing ensemble variance. Default is 'confidence_weighted_var'.
        verbose: Enable detailed progress output. Default is False.

    Returns:
        dict[str, pd.DataFrame]: Dictionary mapping view names to smoothed DataFrames containing
        processed keypoint data with statistics including coordinates, likelihoods, ensemble
        medians, ensemble variances, and posterior variances.
    """
    print(f'Input data loaded for keypoints for multiview data: {keypoint_names}')

    marker_array = input_dfs_to_markerArray(markers_list, keypoint_names, camera_names=views)
    # Uncomment this to skip eks for testing purposes:
    #return {view: markers_list[i][0] for i, view in enumerate(views)}

    # run the ensemble kalman smoother for multiview data
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        marker_array = marker_array,
        keypoint_names = keypoint_names,
        smooth_param = None,
        quantile_keep_pca= quantile_keep_pca, #quantile_keep_pca
        camera_names = views,
        s_frames = [(None,None)], # Keemin wil fix 
        avg_mode = avg_mode,
        var_mode = var_mode,
        inflate_vars = True,
        inflate_vars_kwargs = inflate_vars_kwargs,
        n_latent = n_latent,
        verbose = verbose,
        pca_object = pca_object,
    )

    # Process results for each view
    results_arrays = {}
    results_dfs = {}

    for view_idx, view in enumerate(views):
        if view_idx >= len(camera_dfs) or camera_dfs[view_idx] is None:
            print(f'No results available for view {view}')
            continue
            
        # Extract relevant columns for numpy array
        result_cols = [
            'x', 'y', 'likelihood', 
            'x_ens_median', 'y_ens_median',
            'x_ens_var', 'y_ens_var', 
            'x_posterior_var', 'y_posterior_var'
        ]
        
        df = camera_dfs[view_idx]
        array_data = df.loc[
            :, 
            df.columns.get_level_values(2).isin(result_cols)
        ].to_numpy()
        
        results_arrays[view] = array_data
        results_dfs[view] = df
        
        print(f'Successfully processed view {view}')
    
    return results_dfs


# def post_process_ensemble_labels(
#     cfg_lp: DictConfig,
#     results_dir: str,
#     model_type: str,
#     n_labels: int,
#     seed_range: tuple[int, int],
#     views: List[str], 
#     mode: Literal['ensemble_mean', 'ensemble_median', 'eks_singleview', 'eks_multiview'],
#     inference_dirs: List[str],
#     overwrite: bool,
#     pca_object = None, 
#     fa_object = None
# ) -> None:
#     """Post-process ensemble labels"""

    
#     # Load PCA object if it doesn't exist
#     if pca_object is None:
#         # Check if it exists in the global scope
#         global_pca = globals().get('pca_object')
#         if global_pca is not None:
#             pca_object = global_pca
#             print("Loaded PCA object from global scope")
#         else:
#             # Try to load from the file
#             try:
#                 with open(pca_model_path, "rb") as f:
#                     pca_object = CustomUnpickler(f).load()
#                 print(f"Loaded PCA model from {pca_model_path} and the PCA components shape: {pca_object.components_.shape} ")
#             except Exception as e:
#                 print(f"Could not load PCA model: {e}")
    
    
#     if fa_object is None:
#         global_fa = globals().get('fa_object')
#         if fa_object is not None:
#             fa_object = global_fa
#             print("Loaded FA object from global scope")
#             print("Using FA object for variance inflation")
#         else:
#             # Try to load from the file
#             try:
#                 with open(fa_model_path, "rb") as f:
#                     fa_object = CustomUnpickler(f).load()
#                 print(f"Loaded FA model from {fa_model_path} and the FA components shape: {fa_object.components_.shape} ")
#             except Exception as e:
#                 print(f"Could not load FA model: {e}")


#     # Setup directories
#     base_dir = os.path.dirname(results_dir)
#     ensemble_dir, seed_dirs, _ = setup_ensemble_dirs(
#         base_dir, model_type, n_labels, seed_range, mode, ""
#     )
    
#     for inference_dir in inference_dirs:
#         print(f"Post-processing ensemble predictions for {model_type} {n_labels} {seed_range[0]}-{seed_range[1]} for {inference_dir}")
#         first_seed_dir = seed_dirs[0]
#         inf_dir_in_first_seed = os.path.join(first_seed_dir, inference_dir)
        
#         # Check if the inference_dir inside the first_seed_dir has subdirectories
#         entries = os.listdir(inf_dir_in_first_seed)
#         has_subdirectories = any(os.path.isdir(os.path.join(inf_dir_in_first_seed, entry)) for entry in entries)
        
#         if not has_subdirectories:
#             print(f"No subdirectories found in {inf_dir_in_first_seed}. Skipping.")
#             continue  # Skip this inference directory
            
#         print(f"Subdirectories found in {inf_dir_in_first_seed}. Processing {inference_dir}...")
#         output_dir = os.path.join(ensemble_dir, mode, inference_dir)
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Get original directory structure
#         original_structure = get_original_structure(first_seed_dir, inference_dir, views)
        
#         if mode == 'eks_multiview':
#             # For multiview mode, process each sequence once, not per view
#             # Group directories by sequence (removing the view-specific part)
#             sequences = {}
#             for original_dir in original_structure.keys():
#                 # Extract the sequence name by removing view identifier
#                 curr_view = next((part for part in original_dir.split("_") if part in views), 
#                                original_dir.split("_")[-1])
                
#                 # Create a sequence key by replacing the view with a placeholder
#                 sequence_key = original_dir.replace(curr_view, "VIEW_PLACEHOLDER")
                
#                 if sequence_key not in sequences:
#                     sequences[sequence_key] = []
                
#                 # Store the original directory and its view
#                 sequences[sequence_key].append((original_dir, curr_view))
            
#             # Process each sequence once (not per view)
#             for sequence_key, dir_view_pairs in sequences.items():
#                 # Take the first directory as reference for processing
#                 first_dir, first_view = dir_view_pairs[0]
#                 csv_files = original_structure[first_dir]
                
#                 print(f"Processing sequence: {sequence_key} with {len(csv_files)} CSV files")

#                 # Process this sequence only once
#                 process_multiview_directory(
#                     first_dir, csv_files, first_view, views, 
#                     seed_dirs, inference_dir, output_dir, mode,
#                     pca_object=pca_object,
#                     fa_object=fa_object,
#                     get_bbox_path_fn=lambda p: _get_bbox_path_fn(p, Path(results_dir), Path(cfg_lp.data.data_dir))
#                 )
            
#             # Process final predictions for all views
#             for view in views:
#                 process_final_predictions(
#                     view=view,
#                     output_dir=output_dir,
#                     seed_dirs=seed_dirs,
#                     inference_dir=inference_dir,
#                     cfg_lp=cfg_lp
#                 )
                
#         else:
#             # Process each view separately for other modes
#             for view in views:
#                 print(f"\nProcessing view: {view}")
                
#                 view_dirs = {
#                     dir_name: files 
#                     for dir_name, files in original_structure.items()
#                     if view in dir_name
#                 }
                
#                 # Process each view-specific directory
#                 for original_dir, csv_files in view_dirs.items():
#                     process_singleview_directory(
#                         original_dir, csv_files, view, 
#                         seed_dirs, inference_dir, output_dir, mode
#                     )
                
#                 # Process final predictions for this view
#                 process_final_predictions(
#                     view=view,
#                     output_dir=output_dir,
#                     seed_dirs=seed_dirs,
#                     inference_dir=inference_dir,
#                     cfg_lp=cfg_lp
#                 )

# def process_final_predictions(
#     view: str,
#     output_dir: str,
#     seed_dirs: List[str],
#     inference_dir: str,
#     cfg_lp: DictConfig,
# ) -> None:
#     """Process and save final predictions for a view, computing metrics"""
    
#     # Load original CSV file to get frame paths
#     orig_pred_file = os.path.join(seed_dirs[0], inference_dir, f'predictions_{view}_new.csv')
#     original_df = pd.read_csv(orig_pred_file, header=[0, 1, 2], index_col=0)
#     original_df = io_utils.fix_empty_first_row(original_df)
#     original_index = original_df.index
#     print(f" the original index is ")
#     results_list = []
    
#     # Process each path from original predictions
#     for img_path in original_index:
#         # Get sequence name and base filename
#         sequence_name = img_path.split('/')[1]  # e.g., "180623_000_bot"
#         base_filename = os.path.splitext(os.path.basename(img_path))[0]  # e.g., "img107262"
        
#         # Get the sequence CSV path
#         snippet_file = os.path.join(output_dir, sequence_name, f"{base_filename}.csv")
#         print(f"\nProcessing {img_path}")
#         print(f"Looking for file: {snippet_file}")
        
#         if os.path.exists(snippet_file):
#             try:
#                 # Load the CSV with multi-index columns
#                 snippet_df = pd.read_csv(snippet_file, header=[0, 1, 2], index_col=0)
#                 snippet_df = io_utils.fix_empty_first_row(snippet_df)
#                 print(f"Loaded snippet file with shape: {snippet_df.shape}")
                
#                 # Ensure odd number of frames
#                 assert snippet_df.shape[0] & 1 == 1, f"Expected odd number of frames, got {snippet_df.shape[0]}"
                
#                 # Get center frame index
#                 idx_frame = int(np.floor(snippet_df.shape[0] / 2))
#                 print(f"Extracting center frame at index {idx_frame}")
                
#                 # Extract center frame and set index
#                 center_frame = snippet_df.iloc[[idx_frame]]  # Keep as DataFrame with one row
#                 center_frame.index = [img_path]
#                 results_list.append(center_frame)
                
#                 print(f"Successfully extracted center frame")
                
#             except Exception as e:
#                 print(f"Error processing {snippet_file}: {str(e)}")
#                 print("Full error:")
#                 import traceback
#                 traceback.print_exc()
#         else:
#             print(f"Warning: Could not find file {snippet_file}")
    
#     if results_list:
#         print(f"\nCombining {len(results_list)} processed frames")
#         # Combine all results in original order
#         results_df = pd.concat(results_list)
#         print(f"Combined DataFrame shape: {results_df.shape}")
        
#         # Reindex to match original
#         results_df = results_df.reindex(original_index)
#         print(f"Final DataFrame shape: {results_df.shape}")
        
#         # Add "set" column for labeled data predictions
#         results_df.loc[:,("set", "", "")] = "train"
        
#         # Save predictions
#         preds_file = os.path.join(output_dir, f'predictions_{view}_new.csv')
#         results_df.to_csv(preds_file)
#         print(f"Saved predictions to {preds_file}")
        
#         # Compute metrics
#         cfg_lp_view = cfg_lp.copy()
#         cfg_lp_view.data.csv_file = [f'CollectedData_{view}_new.csv']
#         cfg_lp_view.data.view_names = [view]
        
#         try:
#             compute_metrics(cfg=cfg_lp_view, preds_file=[preds_file], data_module=None) #. Ichanged the preds_file to a list because of a problem... 
#             print(f"Successfully computed metrics for {preds_file}")
#         except Exception as e:
#             print(f"Error computing metrics for {view}: {str(e)}")
#             print(traceback.format_exc())
#     else:
#         print(f"Warning: No frames processed for view {view}")
