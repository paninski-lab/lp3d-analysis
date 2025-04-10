import os
import pandas as pd
import numpy as np
import pickle 
import traceback

from omegaconf import DictConfig
from typing import Optional, Union, Callable, List, Literal, Tuple, Any
from pathlib import Path

from lp3d_analysis import io
from lp3d_analysis.utils import(
    setup_ensemble_dirs,
    get_original_structure,
    add_variance_columns,
    fill_ensemble_results,
    process_predictions,
    process_final_predictions,
)   

from lightning_pose.utils import io as io_utils

from eks.singlecam_smoother import ensemble_kalman_smoother_singlecam, fit_eks_singlecam
#     compute_metrics,
# )

from eks.singlecam_smoother import ensemble_kalman_smoother_singlecam, fit_eks_singlecam
from eks.multicam_smoother import ensemble_kalman_smoother_multicam
from eks.utils import  format_data  # convert_lp_dlc, make_dlc_pandas_index
from eks.core import jax_ensemble
from eks.marker_array import MarkerArray, input_dfs_to_markerArray


# loading pca objects and fa objects if necessart
# pca_model_path = "/teamspace/studios/this_studio/ZZ_pca_objects/pca_object_inD_mirror-mouse-separate.pkl"
# pca_model_path = "/teamspace/studios/this_studio/ZZ_pca_objects/pca_object_inD_fly.pkl"
# pca_model_path = "/teamspace/studios/this_studio/ZZ_pca_objects/pca_object_inD_chickadee-crop_6pcs_new.pkl"

# fa_model_path = "/teamspace/studios/this_studio/ZZ_pca_objects/pca_object_inD_mirror-mouse-separate.pkl"
# fa_model_path = "/teamspace/studios/this_studio/ZZ_pca_objects/fa_object_inD_fly.pkl"
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

# -----------------------------------------------------------------------------
# Functions for dealing with bounding box issues
# -----------------------------------------------------------------------------

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

def generate_cropped_csv_file(
    input_csv_file: str | Path,
    input_bbox_file: str | Path,
    output_csv_file: str | Path,
    img_height: int | None = None,
    img_width: int | None = None,
    mode: str = "subtract",
):
    """Translate a CSV file by bbox file.
    Requires the files have the same index.

    Defaults to subtraction. Can use mode='add' to map from cropped to original space.
    """
    if mode not in ("add", "subtract"):
        raise ValueError(f"{mode} is not a valid mode")
    # Load csv and bbox data   
    csv_data = pd.read_csv(input_csv_file, header=[0, 1, 2], index_col=0)
    csv_data = io_utils.fix_empty_first_row(csv_data)
    bbox_data = pd.read_csv(input_bbox_file, index_col=0)

    for col in csv_data.columns:
        if col[-1] in ("x", "y"):
            vals = csv_data[col].copy()  # Make a copy to avoid modifying the original
            
            if mode == "add":
                # First scale if needed
                if col[-1] == "x" and img_width:
                    vals = (vals / img_width) * bbox_data["w"]
                elif col[-1] == "y" and img_height:
                    vals = (vals / img_height) * bbox_data["h"]
                
                # Then add bbox offset
                csv_data[col] = vals + bbox_data[col[-1]]
                
            elif mode == "subtract":
                # First subtract bbox offset
                vals = vals - bbox_data[col[-1]]
                
                # Then scale if needed
                if col[-1] == "x" and img_width:
                    vals = (vals / bbox_data["w"]) * img_width
                elif col[-1] == "y" and img_height:
                    vals = (vals / bbox_data["h"]) * img_height
                    
                csv_data[col] = vals

    # for col in csv_data.columns:
    #     if col[-1] in ("x", "y"):
    #         vals = csv_data[col]
            
    #         if mode == "add":
    #             if col[-1] == "x" and img_width:
    #                 vals = (vals / img_width) * bbox_data["w"]
    #             elif col[-1] == "y" and img_height:
    #                 vals = (vals / img_height) * bbox_data["h"]
                
    #         if mode == "subtract":
    #             csv_data[col] = vals - bbox_data[col[-1]]
    #         else:
    #             csv_data[col] = vals + bbox_data[col[-1]]

    #         if mode == "subtract":
    #             if col[-1] == "x" and img_width:
    #                 csv_data[col] = (csv_data[col] / bbox_data["w"]) * img_width 
    #             elif col[-1] == "y" and img_height:
    #                 csv_data[col] = (csv_data[col] / bbox_data["h"]) * img_height

    output_csv_file = Path(output_csv_file)
    output_csv_file.parent.mkdir(parents=True, exist_ok=True)
    csv_data.to_csv(output_csv_file)

def prepare_uncropped_csv_files(csv_files: list[str], get_bbox_path_fn: Callable[[Path, ], Path]):
    # Check if a bbox file exists
    # has_checked_bbox_file caches the file check
    has_checked_bbox_file = False
    csv_files_uncropped = []
    for p in csv_files:
        p = Path(p)
        # p is the absolute path to the prediction file.
        # The bbox path has the same relative directory structure but rooted in the data directory
        # and suffixed by _bbox.csv.
        bbox_path = get_bbox_path_fn(p)
        
        if not has_checked_bbox_file:
            if not bbox_path.is_file():
                return None
            has_checked_bbox_file = True

        # If there's a bbox_file, remap to original space by adding bbox df to preds df.
        # Save as _uncropped.csv version of preds_file.
        remapped_p = p.with_stem(p.stem + "_uncropped")
        csv_files_uncropped.append(str(remapped_p))
        generate_cropped_csv_file(p, bbox_path, remapped_p, img_height = 320, img_width = 320, mode="add")

    return csv_files_uncropped

# -----------------------------------------------------------------------------
# Higher-level processing functions 
# -----------------------------------------------------------------------------

def process_multiview_directories_concat(
    views: List[str],
    seed_dirs: List[str],
    inference_dir: str,
    output_dir: str,
    mode: str,
    pca_object = None,
    fa_object = None,
    get_bbox_path_fn: Callable[[Path, ], Path] | None = None,
) -> None:
    """Process multiview directories by concatenating CSV files and applying Multiview EKS."""
    
    # Setup FA parameters for variance inflation if available
    inflate_vars_kwargs = {}
    if fa_object is not None:
        print("Using FA object for variance inflation")
        inflate_vars_kwargs = {
            'loading_matrix': fa_object.components_.T,
            'mean': np.zeros_like(fa_object.mean_)  # Avoid centering twice
        }
    else:
        print("No FA object available for variance inflation")
    
    # Get video directory from first seed
    video_dir = os.path.join(seed_dirs[0], inference_dir)
    sequences = {} # Group directories across views by sequence name

    for view in views:
        view_dirs = [d for d in os.listdir(video_dir) if view in d and os.path.isdir(os.path.join(video_dir, d))]
        print(f"View {view} dirs: {view_dirs}")
        
        for dir_name in view_dirs:
            # Extract sequence name by removing view identifier
            parts = dir_name.split('_')
            sequence_key = '_'.join([p for p in parts if p not in views])
            
            if sequence_key not in sequences:
                sequences[sequence_key] = {}
            sequences[sequence_key][view] = dir_name
    
    # Process each sequence
    for sequence_key, view_dirs in sequences.items():
        all_view_files = {}
        
        # Process each view in the sequence
        for view, dir_name in view_dirs.items():
            print(f"Processing view {view} directory: {dir_name}")
            view_output_path = os.path.join(output_dir, dir_name)
            os.makedirs(view_output_path, exist_ok=True)
            
            # Collect and group CSV files by seed
            csv_files_by_seed = {}
            for seed_idx, seed_dir in enumerate(seed_dirs):
                seed_sequence_dir = os.path.join(seed_dir, inference_dir, dir_name)
                if not os.path.exists(seed_sequence_dir):
                    continue
                    
                csv_files = sorted(
                    [os.path.join(seed_sequence_dir, f) for f in os.listdir(seed_sequence_dir)
                     if f.endswith(".csv") and not f.endswith("_uncropped.csv")],
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace("img", ""))
                )
                
                if csv_files:
                    csv_files_by_seed[seed_idx] = csv_files
            
            # Process each seed's CSV files
            for seed_idx, csv_files in csv_files_by_seed.items():
                # Find bbox files if needed
                bbox_files = []
                if get_bbox_path_fn is not None:
                    bbox_files = [str(get_bbox_path_fn(Path(csv_file))) for csv_file in csv_files]
                    bbox_files = [bf for bf in bbox_files if os.path.exists(bf)]
                    
                    if bbox_files:
                        bbox_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace("img", "").replace("_bbox", "")))
                
                # Get uncropped files (either from original or by adding bbox offsets)
                uncropped_files = prepare_uncropped_csv_files(csv_files, get_bbox_path_fn) if bbox_files else csv_files
                if not uncropped_files:
                    print(f"Warning: No uncropped files for view {view}, seed {seed_idx}")
                    continue
                
                # Concatenate uncropped files
                concat_df = pd.DataFrame()
                for csv_file in uncropped_files:
                    df = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
                    df = io_utils.fix_empty_first_row(df)
                    
                    # Ensure unique indices
                    if not concat_df.empty:
                        # df.index = df.index + concat_df.index[-1] + 1
                        # Calculate the next index value
                        next_index = concat_df.index[-1] + 1
                        df.index = df.index + next_index
                    
                    concat_df = pd.concat([concat_df, df]) if not concat_df.empty else df
                
                if concat_df.empty:
                    continue
                    
                # Save concatenated file
                temp_file = os.path.join(output_dir, f"temp_{view}_{seed_idx}_{sequence_key}.csv")
                concat_df.to_csv(temp_file)
                
                # Process bbox files if available
                if bbox_files:
                    concat_bbox_df = pd.DataFrame()
                    for bbox_file in bbox_files:
                        bbox_df = pd.read_csv(bbox_file)
                        
                        # Ensure unique indices
                        if not concat_bbox_df.empty:
                            # Adjust indices for proper concatenation
                            # for idx_col in ['Unnamed: 0']:
                            #     if idx_col in bbox_df.columns and idx_col in concat_bbox_df.columns:
                            #         bbox_df[idx_col] += concat_bbox_df[idx_col].max() + 1
                            
                            # bbox_df.index = bbox_df.index + concat_bbox_df.index[-1] + 1
                            # Calculate the next index value
                            next_index = concat_bbox_df.index[-1] + 1
                            
                            # Update the index
                            bbox_df.index = bbox_df.index + next_index
                            
                            # If 'Unnamed: 0' column exists, update it consistently with the index
                            if 'Unnamed: 0' in bbox_df.columns and 'Unnamed: 0' in concat_bbox_df.columns:
                                bbox_df['Unnamed: 0'] = bbox_df.index

                        concat_bbox_df = pd.concat([concat_bbox_df, bbox_df]) if not concat_bbox_df.empty else bbox_df
                    
                    bbox_temp_file = os.path.join(output_dir, f"temp_bbox_{view}_{seed_idx}_{sequence_key}.csv")
                    concat_bbox_df.to_csv(bbox_temp_file, index=False)
                
                # Store the temp file for this view
                if view not in all_view_files:
                    all_view_files[view] = []
                all_view_files[view].append(temp_file)
        
        # # Format input data and run EKS multiview
        # input_dfs_list = []
        # keypoint_names = []
        
        # for view, files in all_view_files.items():
        #     if files:
        #         view_dfs, kp_names = format_data(input_source=files, camera_names=[view])
        #         if not keypoint_names:
        #             keypoint_names = kp_names
        #         input_dfs_list.extend(view_dfs)

        # Collect all files from all views
        all_files = []
        for view_files in all_view_files.values():
            all_files.extend(view_files)

        # Format all data together with all camera names
        if all_files:
            all_camera_names = list(all_view_files.keys())
            input_dfs_list, keypoint_names = format_data(
                input_source=all_files,
                camera_names=all_camera_names
            )
        
        if not input_dfs_list:
            print(f"No valid data found for sequence {sequence_key}")
            continue
            
        print(f"Running EKS multiview on {len(input_dfs_list)} DataFrames")
        results_dfs = run_eks_multiview(
            markers_list=input_dfs_list,
            keypoint_names=keypoint_names,
            views=views,
            quantile_keep_pca=50,
            pca_object=pca_object,
            inflate_vars_kwargs=inflate_vars_kwargs
        )
        
        # Save results for each view
        for view, result_df in results_dfs.items():
            view_dir = view_dirs[view]
            view_output_path = os.path.join(output_dir, view_dir)
            result_file = os.path.join(view_output_path, "concatenated_sequence.csv") # output file paths
            
            # Find bbox file if available
            bbox_file = None
            if get_bbox_path_fn is not None:
                seed_idx = list(csv_files_by_seed.keys())[0] if csv_files_by_seed else 0
                bbox_file = os.path.join(output_dir, f"temp_bbox_{view}_{seed_idx}_{sequence_key}.csv")
                if not os.path.exists(bbox_file):
                    bbox_file = None
            
            # Handle with or without bbox files
            if bbox_file:
                # Save uncropped result first, then generate cropped version
                uncropped_file = os.path.join(view_output_path, "concatenated_sequence_uncropped.csv")
                result_df.to_csv(uncropped_file)
                
                generate_cropped_csv_file(
                    input_csv_file=uncropped_file,
                    input_bbox_file=bbox_file,
                    output_csv_file=result_file,
                    img_height=320,
                    img_width=320,
                    mode="subtract"
                )
            else:
                # No bbox file, save result directly
                result_df.to_csv(result_file)
        
        # Clean up temporary files
        for view in views:
            for seed_idx in range(len(seed_dirs)):
                for file_type in ['temp', 'temp_bbox']:
                    temp_file = os.path.join(output_dir, f"{file_type}_{view}_{seed_idx}_{sequence_key}.csv")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

     
def process_singleview_directory(
    original_dir: str,
    csv_files: List[str],
    view: str,
    seed_dirs: List[str],
    inference_dir: str,
    output_dir: str,
    mode: str
) -> None:
    """Process a single view directory"""
    print(f"Processing directory: {original_dir}")
    
    # Identify current view from directory name
    curr_view = next((part for part in original_dir.split("_") if part in [view]), 
                     original_dir.split("_")[-1])
    print(f"Identified view: {curr_view}")
    sequence_output_dir = os.path.join(output_dir, original_dir) # Create output directory matching original structure
    os.makedirs(sequence_output_dir, exist_ok=True)
    
    # Process each CSV file in the directory
    for csv_file in csv_files:
        all_pred_files = []
        curr_dir_name = original_dir.replace(curr_view, view) if curr_view in original_dir else original_dir
        
        for seed_dir in seed_dirs:
            seed_video_dir = os.path.join(seed_dir, inference_dir)
            seed_sequence_dir = os.path.join(seed_video_dir, curr_dir_name)
            
            pred_file = os.path.join(seed_sequence_dir, csv_file)
            if os.path.exists(seed_sequence_dir) and os.path.exists(pred_file):
                all_pred_files.append(pred_file)
        
        if all_pred_files:
            # Process ensemble data
            input_dfs_list, keypoint_names = format_data(
                input_source=all_pred_files,
                camera_names=None,
            )
            
            # Get column structure and process based on mode
            if mode in ['ensemble_mean', 'ensemble_median']:
                results_df = process_ensemble_frames(all_pred_files, keypoint_names, mode=mode)
            elif mode == 'eks_singleview':
                results_df = run_eks_singleview(
                    markers_list=input_dfs_list,
                    keypoint_names=keypoint_names
                )
            
            # Save results
            result_file = os.path.join(sequence_output_dir, csv_file)
            results_df.to_csv(result_file)
            print(f"Saved ensemble {mode} predictions to {result_file}")

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


def post_process_ensemble_labels_concat(
    cfg_lp: DictConfig,
    results_dir: str,
    model_type: str,
    n_labels: int,
    seed_range: tuple[int, int],
    views: List[str], 
    mode: Literal['ensemble_mean', 'ensemble_median', 'eks_singleview', 'eks_multiview'],
    inference_dirs: List[str],
    overwrite: bool,
    pca_object = None, 
    fa_object = None
) -> None:
    """
    Post-process ensemble labels from directories
    
    Args:
        cfg_lp: Configuration dictionary
        results_dir: Base directory for results
        model_type: Type of model
        n_labels: Number of labels
        seed_range: Range of seeds (start, end)
        views: List of view names
        mode: Processing mode
        inference_dirs: List of inference directories
        overwrite: Whether to overwrite existing files
        pca_object: PCA object (optional)
        fa_object: FA object (optional)
    """
    # Load models if needed
    pca_object, fa_object = _load_latent_models(pca_object, fa_object)
    print(f"Using PCA object: {pca_object}")
    print(f"Using FA object: {fa_object}")

    # Setup directories
    base_dir = os.path.dirname(results_dir)
    ensemble_dir, seed_dirs, _ = setup_ensemble_dirs(
        base_dir, model_type, n_labels, seed_range, mode, ""
    )
    
    for inference_dir in inference_dirs:
        print(f"Post-processing ensemble predictions for {model_type} {n_labels} {seed_range[0]}-{seed_range[1]} for {inference_dir}")
        first_seed_dir = seed_dirs[0]
        inf_dir_in_first_seed = os.path.join(first_seed_dir, inference_dir)
        
        # Check if the inference_dir inside the first_seed_dir has subdirectories
        entries = os.listdir(inf_dir_in_first_seed)
        has_subdirectories = any(os.path.isdir(os.path.join(inf_dir_in_first_seed, entry)) for entry in entries)
        
        if not has_subdirectories:
            print(f"No subdirectories found in {inf_dir_in_first_seed}. Skipping.")
            continue
            
        print(f"Subdirectories found in {inf_dir_in_first_seed}. Processing {inference_dir}...")
        output_dir = os.path.join(ensemble_dir, mode, inference_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process using the new concatenation approach
        if mode == 'eks_multiview':
            # Use the new concatenation-based processing
            process_multiview_directories_concat(
                views=views,
                seed_dirs=seed_dirs,
                inference_dir=inference_dir,
                output_dir=output_dir,
                mode=mode,
                pca_object=pca_object,
                fa_object=fa_object,
                get_bbox_path_fn= lambda p: _get_bbox_path_fn(p, Path(results_dir), Path(cfg_lp.data.data_dir)) #None  # Update this if bbox handling is needed
            )
            
            # Process final predictions for all views using the concat approach
            for view in views:
                process_final_predictions(
                    view=view,
                    output_dir=output_dir,
                    seed_dirs=seed_dirs,
                    inference_dir=inference_dir,
                    cfg_lp=cfg_lp,
                    use_concatenated=True
                )
        else:
            original_structure = get_original_structure(first_seed_dir, inference_dir, views)
            print(f"Mode {mode} does not use the concatenation approach. Using original implementation.")
            # Process each view separately for other modes
            for view in views:
                print(f"\nProcessing view: {view}")
                
                view_dirs = {
                    dir_name: files 
                    for dir_name, files in original_structure.items()
                    if view in dir_name
                }
                
                # Process each view-specific directory
                for original_dir, csv_files in view_dirs.items():
                    process_singleview_directory(
                        original_dir, csv_files, view, 
                        seed_dirs, inference_dir, output_dir, mode
                    )
                
                # Process final predictions for this view
                process_final_predictions(
                    view=view,
                    output_dir=output_dir,
                    seed_dirs=seed_dirs,
                    inference_dir=inference_dir,
                    cfg_lp=cfg_lp,
                    use_concatenated=False
                )

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
) -> None:
    """Post-process ensemble videos"""
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
                for view in views: # collect files for all seeds and views
                    print(f"Processing view: {view}")
                    # Get the filename of the prediction file for this session-view pair.
                    session = first_view_session.replace(views[0], view)
                    for seed_dir in seed_dirs:
                        all_pred_files.append(os.path.join(seed_dir, inference_dir, session))

                # Not None when there's bbox files in the dataset, i.e. chickadee-crop.
                all_pred_files_uncropped = prepare_uncropped_csv_files(
                    all_pred_files, 
                    lambda p: _get_bbox_path_fn(p, Path(results_dir), Path(cfg_lp.data.data_dir))
                )

                # Format data and run multiview EKS
                input_dfs_list, keypoint_names = format_data(
                    input_source=all_pred_files_uncropped or all_pred_files,
                    camera_names=views,
                )

                results_dfs = run_eks_multiview(
                    markers_list=input_dfs_list,
                    keypoint_names=keypoint_names,
                    views=views,
                )

                # Save results for each view
                for view in views:
                    session = first_view_session.replace(views[0], view)
                    result_df = results_dfs[view]
                    result_file = Path(output_dir) / session

                    # If cropped dataset, save to a _uncropped file, so that we can later undo the remapping.
                    if all_pred_files_uncropped is not None:
                        uncropped_result_file = result_file.with_stem(result_file.stem + "_uncropped")
                        result_df.to_csv(uncropped_result_file)
                        
                        # Crop back to original coordinate space
                        bbox_path = _get_bbox_path_fn(result_file, Path(results_dir), Path(cfg_lp.data.data_dir))
                        generate_cropped_csv_file(
                            uncropped_result_file, 
                            bbox_path, 
                            result_file, 
                            img_height=320, 
                            img_width=320, 
                            mode="subtract"
                        )
                    else:
                        result_df.to_csv(result_file)

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
    # run the ensemble kalman smoother for multiview data
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        marker_array = marker_array,
        keypoint_names = keypoint_names,
        smooth_param = None,
        quantile_keep_pca= quantile_keep_pca,
        camera_names = views,
        s_frames = [(None,None)], 
        avg_mode = avg_mode,
        var_mode = var_mode,
        inflate_vars = False,
        inflate_vars_kwargs = inflate_vars_kwargs,
        n_latent =6,
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


