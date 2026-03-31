import os
# Configure JAX memory settings before any JAX imports to prevent segmentation faults
# These settings help prevent "Cannot allocate memory" errors during LLVM compilation
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.5')
xla_flags = os.environ.get('XLA_FLAGS', '')
if '--xla_force_host_platform_device_count' not in xla_flags:
    os.environ['XLA_FLAGS'] = f'{xla_flags} --xla_force_host_platform_device_count=1'.strip()

import re
import pandas as pd
import numpy as np
import pickle 
import traceback

from typing import Optional, Union, Callable
from collections import defaultdict


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
from eks.core import ensemble
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
This is loading the pca and FA objects from config files
'''

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

def _load_latent_models_from_config(cfg_lp, pca_object=None, fa_object=None):
    """Helper function to load PCA and FA models from config if not provided"""
    try:
        if pca_object is None and hasattr(cfg_lp, 'latent_models'):
            if hasattr(cfg_lp.latent_models, 'pca_model_path'):
                pca_path = cfg_lp.latent_models.pca_model_path
                if pca_path and pca_path != 'null' and pca_path.lower() != 'null':
                    print(f"Loading PCA model from config path: {pca_path}")
                    pca_object, _ = load_models(pca_path=pca_path)
                else:
                    print("PCA model path is null or empty in config, skipping PCA model loading")
            else:
                print("No PCA model path found in config")
        
        if fa_object is None and hasattr(cfg_lp, 'latent_models'):
            if hasattr(cfg_lp.latent_models, 'fa_model_path'):
                fa_path = cfg_lp.latent_models.fa_model_path
                if fa_path and fa_path != 'null' and fa_path.lower() != 'null':
                    print(f"Loading FA model from config path: {fa_path}")
                    _, fa_object = load_models(fa_path=fa_path)
                else:
                    print("FA model path is null or empty in config, skipping FA model loading")
            else:
                print("No FA model path found in config")
    
    except Exception as e:
        print(f"Error loading latent models from config: {e}")
        print("Continuing without latent models")
    
    return pca_object, fa_object



def _get_bbox_path_fn(p: Path, results_dir: Path, data_dir: Path) -> Path:
    """Given some preds file `p` and `results_dir`, it will return p's
    corresponding bbox path in the data directory."""
    # Gets the current depth in the model directory.
    n_levels_up = len(p.parent.parts) - len(results_dir.parts)
    # Correct the logic for when transforming results of eks multiview back to cropped space.
    # Check if any folder in the path contains "eks" but NOT "singleview"
    # (covers eks_multiview, non_linear_eks, eks_varinf, etc., but NOT eks_singleview)
    if any("eks" in part and "singleview" not in part for part in p.parts):
        n_levels_up -= 1
    # Get `p` relative to model_dir
    model_dir = p.parents[n_levels_up]
    relative_p = Path(p).relative_to(model_dir)
    p_in_data_dir = data_dir / relative_p
    bbox_path = p_in_data_dir.with_stem(p.stem + "_bbox")
    return bbox_path


def _get_eks_calibration_file(session_name_or_dir: str, data_dir: str, views: List[str]) -> Optional[str]:
    """
    Get the calibration TOML file path for a specific session.
    
    Args:
        session_name_or_dir: Directory name or filename containing session name 
                            (e.g., "PRL43_200617_131904_lBack" or "session_Cam-A.csv")
        data_dir: Path to data directory
        views: List of view names to remove from session name
        
    Returns:
        Path to calibration TOML file, or None if not found
    """
    from lp3d_analysis.utils import extract_session_info, extract_sequence_name
    
    # If it's a directory name (no .csv extension), use extract_sequence_name
    # Otherwise, use extract_session_info for filenames
    if session_name_or_dir.endswith('.csv'):
        # It's a filename - extract session name
        session_name, _ = extract_session_info(session_name_or_dir, views)
        base_session_name = session_name.replace('.short', '')
    else:
        # It's a directory name - extract sequence name (removes view and .short)
        base_session_name = extract_sequence_name(session_name_or_dir, views)
    
    # Look for calibration file in data_dir/calibrations/{base_session_name}.toml
    calib_path = Path(data_dir) / "calibrations" / f"{base_session_name}.toml"
    
    if calib_path.exists():
        return str(calib_path)
    
    return None


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
        # Process with ensemble
        stacked_arrays_reshaped = stacked_arrays.transpose(2, 0, 1)  # (5, 601, 90)
        stacked_arrays_reshaped = input_dfs_to_markerArray([stacked_dfs], keypoint_names, camera_names=[""])
        avg_mode = 'mean' if mode == 'ensemble_mean' else 'median'
        
        ensemble_marker_array = ensemble(
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
    mode: str,
    overwrite: bool = False,
) -> None:
    """Process multiple video files for a single view"""
    
    # Get all CSV files from the first seed directory
    first_seed_dir = seed_dirs[0]
    base_files = os.listdir(os.path.join(first_seed_dir, inference_dir))
    view_files = [f for f in base_files if view in f and f.endswith('.csv')] # all csv files with view name 
    # Process each file (representing a different video) separately
    for file_name in view_files:
        # Check if output file already exists
        preds_file = os.path.join(output_dir, file_name)
        if os.path.exists(preds_file) and not overwrite:
            print(f"Skipping {file_name} - output file already exists (overwrite=False)")
            continue
        
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
        
        # Save results using the same filename (preds_file already set above)
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
    fa_object = None,
    non_linear: bool = False,
    output_folder_name: str | None = None,
) -> None:
    """Post-process ensemble videos
    
    Args:
        output_folder_name: Optional custom name for the output folder (e.g., 'non_linear_eks').
                           If None, uses the mode name (e.g., 'eks_multiview').
    """
    print(f"n_latent is: {n_latent} ")
    # Load models if needed
    pca_object, fa_object = _load_latent_models_from_config(cfg_lp, pca_object, fa_object)
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
    folder_name = output_folder_name if output_folder_name is not None else mode
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
        output_dir = os.path.join(ensemble_dir, folder_name, inference_dir)
        os.makedirs(output_dir, exist_ok=True)

        if mode == 'eks_multiview':
            # Get the files in the inference directory partitioned by view.
            files_by_view = io.collect_files_by_token(list(map(Path, entries)), views)
            # Get the files in the inference directory FOR JUST ONE VIEW.
            # There should be one file per session. These we call "first view sessions".
            first_view_sessions = list(map(str, files_by_view[views[0]]))
            for first_view_session in first_view_sessions:
                # Check if all output files already exist for this session
                all_outputs_exist = True
                for view in views:
                    session = first_view_session.replace(views[0], view)
                    result_file = Path(output_dir) / session
                    if not result_file.exists():
                        all_outputs_exist = False
                        break
                
                if all_outputs_exist and not overwrite:
                    print(f"Skipping session {first_view_session} - all output files already exist (overwrite=False)")
                    continue
                
                # Process multiview case for videos
                all_pred_files = []

                # Get calibration file for this session if non_linear is enabled
                calibration_file = None
                if non_linear:
                    calibration_file = _get_eks_calibration_file(
                        first_view_session, 
                        cfg_lp.data.data_dir, 
                        views
                    )
                    if calibration_file:
                        print(f"[EKS] Using nonlinear EKS for session: {first_view_session}")
                    else:
                        print(f"[EKS] WARNING: nonlinear EKS requested but calibration file not found, falling back to linear EKS")

                # Collect files for all views and seeds
                for view in views:
                    print(f"Processing view: {view}")
                    # Get the filename of the prediction file for this session-view pair.
                    session = first_view_session.replace(views[0], view)

                    for seed_dir in seed_dirs:
                        all_pred_files.append(os.path.join(seed_dir, inference_dir, session))

                # Not None when there's bbox files in the dataset, i.e. chickadee-crop.
                all_pred_files_uncropped = prepare_uncropped_csv_files(all_pred_files, lambda p: _get_bbox_path_fn(p, Path(results_dir), Path(cfg_lp.data.data_dir)))
                print(f"all_pred_files_uncropped is {all_pred_files_uncropped}")
                
                input_dfs_list, keypoint_names = format_data(
                    input_source=all_pred_files_uncropped or all_pred_files,
                    camera_names=views,
                )
 
                # print(f" input dfs list: {input_dfs_list}")

                # Run multiview EKS
                # n_latent = n_latent if n_latent is not None else 3
                results_dfs = run_eks_multiview(
                    markers_list=input_dfs_list,
                    keypoint_names=keypoint_names,
                    views=views,
                    quantile_keep_pca = 50, # in general 50
                    inflate_vars_kwargs=inflate_vars_kwargs,
                    n_latent=n_latent,
                    pca_object=pca_object,
                    calibration=calibration_file,
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
                    mode=mode,
                    overwrite=overwrite,
                )


def load_collected_data(
    cfg_lp: DictConfig,
    view: str
) -> Optional[pd.DataFrame]:
    """Load and process CollectedData file for a specific view."""
    collected_data_path = os.path.join(cfg_lp.data.data_dir, f"CollectedData_{view}_new.csv")
    if not os.path.exists(collected_data_path):
        print(f"Warning: CollectedData file not found for {view}")
        return None
    else: 
        df = pd.read_csv(collected_data_path, header=[0, 1, 2], index_col=0)
        df = io_utils.fix_empty_first_row(df)
        print(f"Loaded CollectedData file with shape: {df.shape}")
        return df

def get_valid_indices(collected_df: pd.DataFrame) -> List[str]:
    """Extract valid image indices from collected data."""
    return [
        idx for idx in collected_df.index 
        if idx not in ['bodyparts', 'coords'] 
        and isinstance(idx, str) 
        and '/' in idx
    ]

def group_indices_by_sequence(indices: List[str]) -> Dict[str, List[str]]:
    """Group indices by sequence name."""
    sequences = defaultdict(list)
    for idx in indices:
        parts = idx.split('/')
        if len(parts) >= 2:
            sequences[parts[1]].append(idx)
    return dict(sequences)

def find_prediction_files(
    path_videos_new: str,
    view: str
) -> List[str]:
    """Find prediction files for a specific view."""
    pattern = rf"(?<![A-Z]){re.escape(view)}(?![A-Z])"
    return [
        os.path.join(path_videos_new, filename)
        for filename in os.listdir(path_videos_new)
        if os.path.isfile(os.path.join(path_videos_new, filename))
        and re.search(pattern, filename)
        and 'pixel_error' not in filename
    ]

def extract_frame_numbers(indices: List[str]) -> Dict[str, int]:
    """Extract frame numbers from image paths."""
    img_to_frame_num = {}
    for idx in indices:
        parts = idx.split('/')
        if len(parts) >= 3:
            frame_file = parts[2]
            # Match both .png and .jpg extensions, and handle leading zeros
            frame_match = re.search(r'img(\d+)\.(png|jpg)', frame_file)
            if frame_match:
                img_to_frame_num[idx] = int(frame_match.group(1))
    return img_to_frame_num



def process_numeric_indices(
    pred_df: pd.DataFrame,
    img_to_frame_num: Dict[str, int],
    combined_df: pd.DataFrame
) -> pd.DataFrame:
    """Process prediction file with numeric indices."""
    if not img_to_frame_num:
        print("Warning: img_to_frame_num is empty, cannot map frames")
        return combined_df
    
    sample_frames = list(img_to_frame_num.values())[:5]
    frame_exists = [frame in pred_df.index for frame in sample_frames]
    
    print(f"Sample frame numbers from image paths: {sample_frames}")
    print(f"Prediction file index range: {pred_df.index.min()} to {pred_df.index.max()}")
    print(f"Prediction file has {len(pred_df)} rows")

    if any(frame_exists):
        match_count = 0
        for img_path, frame_num in img_to_frame_num.items():
            if frame_num in pred_df.index:
                combined_df.loc[img_path] = pred_df.loc[frame_num]
                match_count += 1
        print(f"Matched {match_count} frames by direct frame number")
    else:
        print("Frame numbers don't match directly, attempting position-based mapping...")
        print("WARNING: Position-based mapping assumes labeled frames are sequential in the video.")
        print("This may not be correct if labeled frames are sparse or non-sequential.")
        
        frame_nums = sorted(img_to_frame_num.values())
        if len(frame_nums) > 0:
            min_frame = min(frame_nums)
            max_frame = max(frame_nums)
            pred_min = int(pred_df.index.min())
            pred_max = int(pred_df.index.max())
            print(f"Labeled frame range: {min_frame} to {max_frame}, Prediction index range: {pred_min} to {pred_max}")
        
        sorted_pred_indices = sorted(pred_df.index)
        sorted_img_paths = sorted(img_to_frame_num.keys(), key=lambda x: img_to_frame_num[x])
        min_length = min(len(sorted_pred_indices), len(sorted_img_paths))
        
        if min_length == 0:
            print(f"Warning: min_length is 0. img_to_frame_num has {len(img_to_frame_num)} entries, pred_df has {len(pred_df)} rows")
            return combined_df
        
        print(f"Falling back to position-based mapping for {min_length} frames (first {min(min_length, 5)} frame numbers: {[img_to_frame_num[img] for img in sorted_img_paths[:5]]})")
        for i in range(min_length):
            img_path = sorted_img_paths[i]
            pred_idx = sorted_pred_indices[i]
            combined_df.loc[img_path] = pred_df.loc[pred_idx]
        print(f"Mapped {min_length} frames by position")
    
    return combined_df

def process_string_indices(
    pred_df: pd.DataFrame,
    img_to_frame_num: Dict[str, int],
    combined_df: pd.DataFrame,
    sequence: str
) -> pd.DataFrame:
    """Process prediction file with string indices."""
    seq_indices = [idx for idx in pred_df.index if isinstance(idx, str) and sequence in idx]
    if seq_indices:
        match_count = 0
        for img_path, frame_num in img_to_frame_num.items():
            for pred_idx in seq_indices:
                # Match both .png and .jpg extensions
                pred_frame_match = re.search(r'img(\d+)\.(png|jpg)', pred_idx)
                if pred_frame_match and int(pred_frame_match.group(1)) == frame_num:
                    combined_df.loc[img_path] = pred_df.loc[pred_idx]
                    match_count += 1
                    break
        print(f"Matched {match_count} frames by frame number in path")
    else:
        print("Warning: Could not determine how to map indices for this file")
    return combined_df

def process_prediction_file(
    pred_file: str,
    collected_df: pd.DataFrame,
    img_to_frame_num: Dict[str, int],
    combined_df: Optional[pd.DataFrame],
    sequence: str
) -> Optional[pd.DataFrame]:
    """Process a single prediction file and update combined DataFrame."""
    try:
        pred_df = pd.read_csv(pred_file, header=[0, 1, 2], index_col=0)
        pred_df = io_utils.fix_empty_first_row(pred_df)
        print(f"Loaded prediction file with shape: {pred_df.shape}")

        if combined_df is None:
            combined_df = pd.DataFrame(index=collected_df.index, columns=pred_df.columns)
            print(f"Initialized combined DataFrame with shape: {combined_df.shape}")

        # Check if img_to_frame_num is empty
        if not img_to_frame_num:
            print(f"Warning: img_to_frame_num is empty for sequence {sequence}. Cannot map frames.")
            return combined_df

        if pred_df.index.dtype == 'int64' or all(isinstance(x, int) for x in pred_df.index if isinstance(x, (int, float))):
            return process_numeric_indices(pred_df, img_to_frame_num, combined_df)
        else:
            return process_string_indices(pred_df, img_to_frame_num, combined_df, sequence)

    except Exception as e:
        print(f"Error processing prediction file {pred_file}: {e}")
        print(traceback.format_exc())
        return combined_df

def save_and_evaluate_results(
    combined_df: pd.DataFrame,
    output_dir: str,
    view: str,
    cfg_lp: DictConfig
) -> None:
    """Save combined predictions and compute metrics."""
    if combined_df is None:
        print(f"Warning: No data was combined for {view}")
        return

    combined_df.loc[:, ("set", "", "")] = "train"
    non_empty_count = combined_df.dropna(how='all').shape[0]
    print(f"\nSuccessfully filled {non_empty_count} out of {combined_df.shape[0]} rows")

    output_file = os.path.join(output_dir, f"predictions_{view}_new.csv")
    combined_df.to_csv(output_file)
    print(f"Saved predictions to {output_file}")

    cfg_lp_view = cfg_lp.copy()
    cfg_lp_view.data.csv_file = [f'CollectedData_{view}_new.csv']
    cfg_lp_view.data.view_names = [view]

    try:
        compute_metrics(cfg=cfg_lp_view, preds_file=[output_file], data_module=None)
        print(f"Successfully computed metrics for {view}")
    except Exception as e:
        print(f"Error computing metrics for {view}: {str(e)}")
        print(traceback.format_exc())

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
    
    # Use the specified inference_dir, but if it's "videos_new" and "videos-full_new" exists, prefer that
    specified_inference_path = os.path.join(method_dir, inference_dir)
    videos_full_new_path = os.path.join(method_dir, "videos-full_new")
    
    if inference_dir == "videos_new" and os.path.exists(videos_full_new_path):
        path_videos_new = videos_full_new_path
        print(f"Using videos-full_new directory instead of videos_new: {path_videos_new}")
    else:
        path_videos_new = specified_inference_path
        print(f"Using specified inference directory: {path_videos_new}")
    
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
        
        # Load collected data
        collected_df = load_collected_data(cfg_lp, view)
        if collected_df is None:
            continue

        # Get valid indices and sequences
        valid_indices = get_valid_indices(collected_df)
        sequences = group_indices_by_sequence(valid_indices)
        prediction_files = find_prediction_files(path_videos_new, view)
        print(f"Found {len(prediction_files)} prediction files for {view}")

        # Initialize combined DataFrame
        combined_df = None

        # Process each sequence
        for sequence, indices in sequences.items():
            print(f"\nProcessing sequence: {sequence}")
            
            sequence_pred_files = [f for f in prediction_files if sequence in os.path.basename(f)]
            print(f"Found {len(sequence_pred_files)} prediction files for sequence {sequence}")

            if not sequence_pred_files:
                print(f"No prediction files found for sequence {sequence}")
                continue

            img_to_frame_num = extract_frame_numbers(indices)
            print(f"Extracted {len(img_to_frame_num)} frame numbers from {len(indices)} indices")
            
            if not img_to_frame_num:
                print(f"Warning: Could not extract frame numbers from indices for sequence {sequence}")
                print(f"Sample indices: {indices[:3] if len(indices) >= 3 else indices}")
                continue
            
            for pred_file in sequence_pred_files:
                print(f"Processing file: {os.path.basename(pred_file)}")
                combined_df = process_prediction_file(pred_file, collected_df, img_to_frame_num, combined_df, sequence)

        # Save and evaluate results
        save_and_evaluate_results(combined_df, output_dir, view, cfg_lp)

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
    inflate_vars_kwargs: dict = {},
    verbose: bool = False,
    n_latent: int =3,
    pca_object = None,
    calibration: Optional[str] = None,
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
        calibration: Path to .toml calibration file for nonlinear EKS. If None, uses linear EKS.

    Returns:
        dict[str, pd.DataFrame]: Dictionary mapping view names to smoothed DataFrames containing
        processed keypoint data with statistics including coordinates, likelihoods, ensemble
        medians, ensemble variances, and posterior variances.
    """
    print(f'Input data loaded for keypoints for multiview data: {keypoint_names}')

    marker_array = input_dfs_to_markerArray(markers_list, keypoint_names, camera_names=views)
    # Uncomment this to skip eks for testing purposes:
    #return {view: markers_list[i][0] for i, view in enumerate(views)}

    # Load calibration if provided for nonlinear EKS
    camgroup = None
    if calibration is not None:
        from aniposelib.cameras import CameraGroup
        try:
            camgroup = CameraGroup.load(calibration)
        except Exception as e:
            print(f"[EKS] ERROR: Could not load calibration file, falling back to linear EKS")
            camgroup = None

    # run the ensemble kalman smoother for multiview data
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        marker_array = marker_array,
        keypoint_names = keypoint_names,
        smooth_param = None, # 10000, #None,
        quantile_keep_pca= quantile_keep_pca, 
        camera_names = views,
        # s_frames = [(None,None)], # Keemin wil fix 
        s_frames = [(0,400)], # used to have 10000 
        avg_mode = avg_mode,
        var_mode = var_mode,
        inflate_vars = True,
        inflate_vars_kwargs = inflate_vars_kwargs,
        n_latent = n_latent,
        verbose = verbose,
        pca_object = pca_object,
        camgroup = camgroup,
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

