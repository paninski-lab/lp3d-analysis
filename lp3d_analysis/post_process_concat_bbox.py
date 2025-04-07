import os
import pandas as pd
import numpy as np
import pickle 

from typing import Optional, Union, Callable

from lp3d_analysis import io
from omegaconf import DictConfig
from typing import List, Literal, Tuple, Dict, Any 
from pathlib import Path

from lightning_pose.utils import io as io_utils
# from lightning_pose.utils.cropzoom import generate_cropped_csv_file
from lightning_pose.utils.scripts import (
    compute_metrics,
)

from eks.singlecam_smoother import ensemble_kalman_smoother_singlecam, fit_eks_singlecam
from eks.multicam_smoother import ensemble_kalman_smoother_multicam
from eks.utils import convert_lp_dlc, format_data, make_dlc_pandas_index
from eks.core import jax_ensemble
from eks.marker_array import MarkerArray, input_dfs_to_markerArray

# from lp3d_analysis.pca_global import ensemble_kalman_smoother_multicam2 

'''
This is loading the pca and FA objects - we will probably not use it like that later 
'''

# pca_model_path = "/teamspace/studios/this_studio/ZZ_pca_objects/pca_object_inD_mirror-mouse-separate.pkl"
# pca_model_path = "/teamspace/studios/this_studio/ZZ_pca_objects/pca_object_inD_fly.pkl"
# pca_model_path = "/teamspace/studios/this_studio/ZZ_pca_objects/pca_object_inD_chickadee-crop_6pcs_new.pkl"

# fa_model_path = "/teamspace/studios/this_studio/ZZ_pca_objects/pca_object_inD_mirror-mouse-separate.pkl"
# fa_model_path = "/teamspace/studios/this_studio/ZZ_pca_objects/fa_object_inD_fly.pkl"
# fa_model_path = "/teamspace/studios/this_studio/ZZ_pca_objects/pca_object_inD_chickadee-crop_6pcs_new.pkl"

# #Custom function to force pickle to find NaNPCA2 in pca_global
# class CustomUnpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if name == "NaNPCA":
#             from lightning_pose.utils.pca import  NaNPCA
#             return NaNPCA
#         elif name == "EnhancedFactorAnalysis":
#             from lp3d_analysis.pca_global import EnhancedFactorAnalysis
#             return EnhancedFactorAnalysis
#         return super().find_class(module, name)

# #Load PCA object before defining functions
# try:
#     with open(pca_model_path, "rb") as f:
#         pca_object = CustomUnpickler(f).load()
#     print(f"PCA model loaded successfully from {pca_model_path}.")
# except AttributeError as e:
#     print(f"Error loading PCA model: {e}. Ensure NaNPCA2 is correctly imported from pca_global.py.")
# except FileNotFoundError as e:
#     print(f"Skipping loading pca_object from {pca_model_path}:")
#     print(e)

# try:
#     with open(fa_model_path, "rb") as f:
#         fa_object = CustomUnpickler(f).load()
#     print(f"PCA model loaded successfully from {fa_model_path}. for the purpose of variance inflation.")
# except AttributeError as e:
#     print(f"Error loading PCA model: {e}. Ensure NaNPCA2 is correctly imported from pca_global.py.")
# except FileNotFoundError as e: 
#     print(f"Skipping loading pca_object from {fa_model_path}:")
#     print(e)

# #Load FA object before defining functions
# try:
#     with open(fa_model_path, "rb") as f:
#         fa_object = CustomUnpickler(f).load()
#     print(f"FA model loaded successfully from {fa_model_path}.")
# except AttributeError as e:
#     print(f"Error loading FA model: {e}. Ensure EnhancedFactorAnalysis is correctly imported from pca_global.py.")

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
    # Read csv file from pose_model.cfg.data.csv_file
    # TODO: reuse header_rows logic from datasets.py
    csv_data = pd.read_csv(input_csv_file, header=[0, 1, 2], index_col=0)
    csv_data = io_utils.fix_empty_first_row(csv_data)

    bbox_data = pd.read_csv(input_bbox_file, index_col=0)

    for col in csv_data.columns:
        if col[-1] in ("x", "y"):
            vals = csv_data[col]
            
            if mode == "add":
                if col[-1] == "x" and img_width:
                    vals = (vals / img_width) * bbox_data["w"]
                elif col[-1] == "y" and img_height:
                    vals = (vals / img_height) * bbox_data["h"]
                
            if mode == "subtract":
                csv_data[col] = vals - bbox_data[col[-1]]
            else:
                csv_data[col] = vals + bbox_data[col[-1]]

            if mode == "subtract":
                if col[-1] == "x" and img_width:
                    csv_data[col] = (csv_data[col] / bbox_data["w"]) * img_width 
                elif col[-1] == "y" and img_height:
                    csv_data[col] = (csv_data[col] / bbox_data["h"]) * img_height

    output_csv_file = Path(output_csv_file)
    output_csv_file.parent.mkdir(parents=True, exist_ok=True)
    csv_data.to_csv(output_csv_file)


def process_predictions(pred_file: str, include_likelihood = True, include_variance = False, include_posterior_variance = False, column_structure=None):
    """
    Process predictions from a CSV file and return relevant data structures.
    
    Args:
        pred_file (str): Path to the prediction CSV file
        include_likelihood (bool): Whether to include likelihood values in the selected coordinates 
        include_variance (bool): Whether to include ensemble variances in the selected coordinates
        column_structure: Existing column structure (if any)
        
    Returns:
        tuple: (column_structure, array_data, numeric_cols, keypoint_names, df_index)
               Returns (None, None, None, None, None) if file doesn't exist
    """
    keypoint_names = []  # Initialize keypoint_names 
    
    if not os.path.exists(pred_file):
        print(f"Warning: Could not find predictions file: {pred_file}")
        return None, None, None, None, None
        
    df = pd.read_csv(pred_file, header=[0, 1, 2], index_col=0)
    df = io_utils.fix_empty_first_row(df)  # Add this line

    # Select column structure: Default includes 'x', 'y', and 'likelihood' and I want to load ensemble variances if I can 
    # selected_coords = ['x', 'y', 'likelihood'] if include_likelihood else ['x', 'y']
    selected_coords = ['x', 'y'] # check that it works  really 
    if include_likelihood:
        selected_coords.append('likelihood')
    if include_variance:
        selected_coords.extend(['x_ens_var', 'y_ens_var'])
    elif include_posterior_variance:
        selected_coords.extend(['x_posterior_var', 'y_posterior_var'])
    
    if column_structure is None:
        column_structure = df.loc[:, df.columns.get_level_values(2).isin(selected_coords)].columns
    keypoint_names = list(dict.fromkeys(column_structure.get_level_values(1)))
    print(f'Keypoint names are {keypoint_names}')
    model_name = df.columns[0][0]
    numeric_cols = df.loc[:, column_structure]
    print(f"numeric_cols are {numeric_cols}")
    array_data = numeric_cols.to_numpy()
    
    return column_structure, array_data, numeric_cols, keypoint_names, df.index



def process_final_predictions_concat(
    view: str,
    output_dir: str,
    seed_dirs: List[str],
    inference_dir: str,
    cfg_lp: DictConfig,
) -> None:
    """Process and save final predictions by extracting center frames from concatenated snippets"""
    
    # Load original CSV file to get frame paths
    orig_pred_file = os.path.join(seed_dirs[0], inference_dir, f'predictions_{view}_new.csv')
    original_df = pd.read_csv(orig_pred_file, header=[0, 1, 2], index_col=0)
    original_df = io_utils.fix_empty_first_row(original_df)
    original_index = original_df.index
    print(f"Original index has {len(original_index)} entries")
    
    results_list = []
    
    # Group image paths by their directories
    image_paths_by_dir = {}
    for img_path in original_index:
        # Get sequence name (directory)
        sequence_name = img_path.split('/')[1]  # e.g., "180623_000_bot"
        if sequence_name not in image_paths_by_dir:
            image_paths_by_dir[sequence_name] = []
        image_paths_by_dir[sequence_name].append(img_path)
    
    # Process each directory's frames together
    for sequence_name, img_paths in image_paths_by_dir.items():
        print(f"\nProcessing sequence: {sequence_name} with {len(img_paths)} frames")
        
        # Find the appropriate directory name pattern for this sequence
        # Check different possible directory structures
        possible_dirs = [
            sequence_name,
            f"{sequence_name}_{view}",
            view + "_" + sequence_name
        ]
        
        sequence_dir = None
        for dir_pattern in possible_dirs:
            matching_dirs = [d for d in os.listdir(output_dir) if dir_pattern in d]
            if matching_dirs:
                sequence_dir = matching_dirs[0]
                break
        
        if not sequence_dir:
            print(f"Warning: Could not find directory for sequence {sequence_name}")
            continue
        
        concat_file = os.path.join(output_dir, sequence_dir, "concatenated_sequence.csv")
        print(f"Looking for concatenated file: {concat_file}")
        
        if os.path.exists(concat_file):
            try:
                # Load the concatenated CSV with multi-index columns
                concat_df = pd.read_csv(concat_file, header=[0, 1, 2], index_col=0)
                concat_df = io_utils.fix_empty_first_row(concat_df)
                print(f"Loaded concatenated file with shape: {concat_df.shape}")
                
                # Get the first seed directory with this sequence
                first_seed_dir = None
                for seed_dir in seed_dirs:
                    seed_sequence_dir = os.path.join(seed_dir, inference_dir, sequence_dir)
                    if os.path.exists(seed_sequence_dir):
                        first_seed_dir = seed_sequence_dir
                        break
                
                if not first_seed_dir:
                    print(f"Warning: Could not find seed directory for {sequence_dir}")
                    continue
                
                # Get list of CSV files in order - these represent the original snippets
                csv_files = [
                    f for f in os.listdir(first_seed_dir)
                    if f.endswith(".csv") and not f.endswith("_uncropped.csv")
                ]
                
                # Sort CSV files by their frame numbers
                csv_files.sort(key=lambda x: int(os.path.splitext(x)[0].replace("img", "")))
                print(f"Found {len(csv_files)} CSV files representing snippets")
                
                # Map each original image path to its position in the original sequence of files
                img_to_position = {}
                ordered_img_paths = []
                
                for i, csv_file in enumerate(csv_files):
                    base_name = os.path.splitext(csv_file)[0]
                    img_file = base_name + ".png"
                    img_path = f"labeled-data/{sequence_name}/{img_file}"
                    if img_path in img_paths:
                        img_to_position[img_path] = i
                        ordered_img_paths.append(img_path)
                
                print(f"Mapped {len(img_to_position)} image paths to their positions")
                
                # Sort img_paths by their position in the sequence
                img_paths_ordered = sorted(img_paths, key=lambda x: img_to_position.get(x, float('inf')))
                
                # Process each snippet (51 frames each) and extract the center frame
                snippet_length = 51  # Typical snippet length
                
                for i, img_path in enumerate(img_paths_ordered):
                    # Calculate the center frame index for this snippet
                    snippet_start = i * snippet_length
                    center_idx = snippet_start + (snippet_length // 2)
                    
                    if center_idx < concat_df.shape[0]:
                        # Extract the center frame and set index
                        center_frame = concat_df.iloc[[center_idx]]
                        center_frame.index = [img_path]
                        results_list.append(center_frame)
                        print(f"Extracted center frame at index {center_idx} for snippet {i+1} ({img_path})")
                    else:
                        print(f"Warning: Center index {center_idx} out of bounds for {img_path}")
                
            except Exception as e:
                print(f"Error processing {concat_file}: {str(e)}")
                print("Full error:")
                import traceback
                traceback.print_exc()
        else:
            print(f"Warning: Could not find concatenated file {concat_file}")
    
    if results_list:
        print(f"\nCombining {len(results_list)} processed frames")
        # Combine all results in original order
        results_df = pd.concat(results_list)
        print(f"Combined DataFrame shape: {results_df.shape}")
        
        # Reindex to match original
        results_df = results_df.reindex(original_index)
        print(f"Final DataFrame shape: {results_df.shape}")
        
        # Add "set" column for labeled data predictions
        results_df.loc[:,("set", "", "")] = "train"
        
        # Save predictions
        preds_file = os.path.join(output_dir, f'predictions_{view}_new.csv')
        results_df.to_csv(preds_file)
        print(f"Saved predictions to {preds_file}")
        
        # Compute metrics
        cfg_lp_view = cfg_lp.copy()
        cfg_lp_view.data.csv_file = [f'CollectedData_{view}_new.csv']
        cfg_lp_view.data.view_names = [view]
        
        try:
            compute_metrics(cfg=cfg_lp_view, preds_file=[preds_file], data_module=None)
            print(f"Successfully computed metrics for {preds_file}")
        except Exception as e:
            print(f"Error computing metrics for {view}: {str(e)}")
            print(traceback.format_exc())
    else:
        print(f"Warning: No frames processed for view {view}")


def setup_ensemble_dirs(
    base_dir: str, 
    model_type: str, 
    n_labels: int, 
    seed_range: tuple[int, int],
    mode: str,
    inference_dir: str
) -> Tuple[str, List[str], str]:
    """Set up and return directories for ensemble processing"""
    ensemble_dir = os.path.join(
        base_dir,
        f"{model_type}_{n_labels}_{seed_range[0]}-{seed_range[1]}"
    )
    seed_dirs = [
        os.path.join(base_dir, f"{model_type}_{n_labels}_{seed}")
        for seed in range(seed_range[0], seed_range[1] + 1)
    ]
    output_dir = os.path.join(ensemble_dir, mode, inference_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    return ensemble_dir, seed_dirs, output_dir

def get_original_structure(
    first_seed_dir: str, 
    inference_dir: str, 
    views: List[str]
) -> Dict[str, List[str]]:
    """Create a mapping of original directories and their files"""
    video_dir = os.path.join(first_seed_dir, inference_dir)
    original_structure = {}
    
    for view in views:
        view_dirs = [d for d in os.listdir(video_dir) if view in d]
        print(f"View {view} dirs: {view_dirs}")
        
        for dir_name in view_dirs:
            dir_path = os.path.join(video_dir, dir_name)
            # Ensure dir_path is a directory before listing its contents
            if not os.path.isdir(dir_path):
                print(f"Skipping {dir_path}, as it is not a directory.")
                continue

            print(f"Checking directory: {dir_path}")
            csv_files = [
                f
                for f in os.listdir(dir_path)
                if f.endswith(".csv") and not f.endswith("_uncropped.csv")
            ]
            print(f"Found {len(csv_files)} CSV files in {dir_name}")
            original_structure[dir_name] = csv_files
        
            
    return original_structure


def process_ensemble_frames(
    all_pred_files: List[str],
    keypoint_names: List[str],
    views: List[str] = None,
    mode: str = "ensemble_mean"
) -> pd.DataFrame:
    """Process ensemble frames using the specified mode"""
    
    # Get input data in the right format
    if mode == 'eks_multiview' and views:
        input_dfs_list, keypoint_names = format_data(
            input_source=all_pred_files,
            camera_names=views,
        )
        
        results_dfs = run_eks_multiview(
            markers_list=input_dfs_list,
            keypoint_names=keypoint_names,
            views=views,
        )
        
        # For multiview, we return a dict of dataframes, one per view
        return results_dfs
    else:
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
            
            # Create results DataFrame
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
            
        elif mode == 'eks_singleview':
            results_df = run_eks_singleview(
                markers_list=stacked_dfs,
                keypoint_names=keypoint_names
            )
            
            # Update column structure based on results
            column_structure = results_df.columns[
                results_df.columns.get_level_values(2).isin([
                    'x', 'y', 'likelihood', 'x_ens_median', 'y_ens_median',
                    'x_ens_var', 'y_ens_var', 'x_posterior_var', 'y_posterior_var'
                ])
            ]
        else:
            raise ValueError(f"Invalid mode: {mode}")
            
        return results_df

def add_variance_columns(column_structure):
    """Add variance columns to the column structure"""
    current_tuples = list(column_structure)
    
    # Add variance columns for each bodypart
    new_tuples = []
    for scorer, bodypart, coord in current_tuples:
        new_tuples.append((scorer, bodypart, coord))
        if coord == 'likelihood':  # After each likelihood, add variance columns
            new_tuples.append((scorer, bodypart, 'x_ens_var'))
            new_tuples.append((scorer, bodypart, 'y_ens_var'))
    
    # Create new column structure with added variance columns
    return pd.MultiIndex.from_tuples(new_tuples, names=['scorer', 'bodyparts', 'coords'])

def fill_ensemble_results(
    results_df: pd.DataFrame, 
    ensemble_preds: np.ndarray, 
    ensemble_vars: np.ndarray, 
    ensemble_likes: np.ndarray, 
    keypoint_names: List[str], 
    column_structure: pd.MultiIndex
) -> None:
    """Fill in ensemble results into the DataFrame"""
    for k, bp in enumerate(keypoint_names):
        # Use scorer from column_structure if not in bp; assume first scorer applies
        scorer = bp.split('/', 1)[0] if '/' in bp else column_structure.levels[0][0]
        bodypart = bp.split('/', 1)[1] if '/' in bp else bp

        # Fill coordinates and likelihood
        results_df.loc[:, (scorer, bodypart, 'x')] = ensemble_preds[:, k, 0]
        results_df.loc[:, (scorer, bodypart, 'y')] = ensemble_preds[:, k, 1]
        results_df.loc[:, (scorer, bodypart, 'likelihood')] = ensemble_likes[:, k]
        
        # Add ensemble variances
        results_df.loc[:, (scorer, bodypart, 'x_ens_var')] = ensemble_vars[:, k, 0]
        results_df.loc[:, (scorer, bodypart, 'y_ens_var')] = ensemble_vars[:, k, 1]



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


def prepare_uncropped_csv_files(csv_files: list[str], get_bbox_path_fn: Callable[[Path, ], Path]):
    # Check if a bbox file exists
    # has_checked_bbox_file caches the file check
    has_checked_bbox_file = False
    csv_files_uncropped = []
    for p in csv_files:
        p = Path(p)
        print(f"p is {p}")
        # p is the absolute path to the prediction file.
        # The bbox path has the same relative directory structure but rooted in the data directory
        # and suffixed by _bbox.csv.
        bbox_path = get_bbox_path_fn(p)
        print(f"bbox_path is {bbox_path}")

        if not has_checked_bbox_file:
            if not bbox_path.is_file():
                return None
            has_checked_bbox_file = True

        # If there's a bbox_file, remap to original space by adding bbox df to preds df.
        # Save as _uncropped.csv version of preds_file.
        remapped_p = p.with_stem(p.stem + "_uncropped")
        print(f"remapped_p is {remapped_p}")
        csv_files_uncropped.append(str(remapped_p))
        generate_cropped_csv_file(p, bbox_path, remapped_p, img_height = 320, img_width = 320, mode="add")

    return csv_files_uncropped

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
    """Process multiview directories by concatenating CSV files and applying EKS."""
    
    # Setup FA parameters for variance inflation if available
    inflate_vars_kwargs = {}
    if fa_object is not None:
        print("Using FA object for variance inflation")
        try:
            inflate_vars_kwargs = {
                'loading_matrix': fa_object.components_.T,
                'mean': np.zeros_like(fa_object.mean_)  # Avoid centering twice
            }
        except AttributeError as e:
            print(f"Error extracting FA parameters: {e}")
    else:
        print("No FA object available for variance inflation")
    
    # Get video directory from first seed
    video_dir = os.path.join(seed_dirs[0], inference_dir)
    
    # Group directories across views by sequence name
    sequences = {}
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
        print(f"\nProcessing sequence: {sequence_key}")
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
                    
                csv_files = [
                    os.path.join(seed_sequence_dir, f) 
                    for f in os.listdir(seed_sequence_dir)
                    if f.endswith(".csv") and not f.endswith("_uncropped.csv")
                ]
                
                if csv_files:
                    csv_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace("img", "")))
                    csv_files_by_seed[seed_idx] = csv_files
            
            # Process each seed's CSV files
            for seed_idx, csv_files in csv_files_by_seed.items():
                # Get corresponding bbox files if needed
                bbox_files = []
                if get_bbox_path_fn is not None:
                    bbox_files = []
                    for csv_file in csv_files:
                        try:
                            bbox_path = get_bbox_path_fn(Path(csv_file))
                            if os.path.exists(bbox_path):
                                bbox_files.append(str(bbox_path))
                        except Exception as e:
                            print(f"Error finding bbox for {csv_file}: {e}")
                    
                    if bbox_files:
                        bbox_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace("img", "").replace("_bbox", "")))
                
                # Generate or use uncropped files
                uncropped_files = prepare_uncropped_csv_files(csv_files, get_bbox_path_fn ) if bbox_files else csv_files # only if we have bbox files 
                if not uncropped_files:
                    print(f"Warning: No uncropped files for view {view}, seed {seed_idx}")
                    continue
                
                # Concatenate uncropped files
                concat_df = pd.DataFrame()
                for csv_file in uncropped_files:
                    try:
                        df = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
                        df = io_utils.fix_empty_first_row(df)
                        
                        # Ensure unique indices
                        if not concat_df.empty:
                            df.index = df.index + concat_df.index[-1] + 1
                        
                        concat_df = pd.concat([concat_df, df]) if not concat_df.empty else df
                    except Exception as e:
                        print(f"Error reading {csv_file}: {e}")
                
                if concat_df.empty:
                    continue
                    
                # Save concatenated file
                temp_file = os.path.join(output_dir, f"temp_{view}_{seed_idx}_{sequence_key}.csv")
                concat_df.to_csv(temp_file)
                csv_files_by_seed[seed_idx] = [temp_file]
                
                # Process bbox files if available
                if bbox_files:
                    concat_bbox_df = pd.DataFrame()
                    for bbox_file in bbox_files:
                        try:
                            # bbox_df = pd.read_csv(bbox_file, index_col=0)
                            bbox_df = pd.read_csv(bbox_file)
                            
                            # Ensure unique indices
                            if not concat_bbox_df.empty:
                                # Adjust indices for proper concatenation
                                for idx_col in ['Unnamed: 0']:
                                    if idx_col in bbox_df.columns and idx_col in concat_bbox_df.columns:
                                        max_val = concat_bbox_df[idx_col].max()
                                        bbox_df[idx_col] = bbox_df[idx_col] + max_val + 1
                                
                                bbox_df.index = bbox_df.index + concat_bbox_df.index[-1] + 1
                            
                            concat_bbox_df = pd.concat([concat_bbox_df, bbox_df]) if not concat_bbox_df.empty else bbox_df
                        except Exception as e:
                            print(f"Error reading bbox file {bbox_file}: {e}")
                    
                    if not concat_bbox_df.empty:
                        bbox_temp_file = os.path.join(output_dir, f"temp_bbox_{view}_{seed_idx}_{sequence_key}.csv")
                        concat_bbox_df.to_csv(bbox_temp_file, index=False)
            
                if view not in all_view_files:
                    all_view_files[view] = []
                all_view_files[view].append(temp_file)

            # # Collect all concatenated files for this view
            # view_files = []
            # for seed_idx in csv_files_by_seed.keys():
            #     temp_file = os.path.join(output_dir, f"temp_{view}_{seed_idx}_{sequence_key}.csv")
            #     if os.path.exists(temp_file):
            #         view_files.append(temp_file)
            
            # if view_files:
            #     all_view_files[view] = view_files
        
        
        # Format input data from concatenated files
        print(f"Preparing input data for EKS multiview across {len(all_view_files)} views")
        # Format input data
        input_dfs_list = []
        keypoint_names = []
        
        for view, files in all_view_files.items():
            if files:
                view_dfs, kp_names = format_data(input_source=files, camera_names=[view])
                if not keypoint_names:
                    keypoint_names = kp_names
                input_dfs_list.extend(view_dfs)
        
            
        print(f"Running EKS multiview on {len(input_dfs_list)} DataFrames for sequence {sequence_key}")
        print(f" input _dfs_list: {input_dfs_list}")
        # Run EKS multiview
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
            os.makedirs(view_output_path, exist_ok=True)

            # Define result file paths without sequence key in the name
            result_file = os.path.join(view_output_path, "concatenated_sequence.csv") 

            # since bbox files are the same across seeds, we can just use the one from the first seed 
            bbox_file = None
            if get_bbox_path_fn is not None:
                # Use just the first seed's bbox file since all are the same
                seed_idx = list(csv_files_by_seed.keys())[0] if csv_files_by_seed else 0
                temp_bbox_file = os.path.join(output_dir, f"temp_bbox_{view}_{seed_idx}_{sequence_key}.csv")

                if os.path.exists(temp_bbox_file):
                    bbox_file = temp_bbox_file
                    print(f"Using bbox file: {bbox_file}")
            
            if bbox_file:
                # We have a bbox file, so we need to save the uncropped result first
                uncropped_result_file = os.path.join(view_output_path, "concatenated_sequence_uncropped.csv")
                print(f"Saving uncropped result for view {view} to {uncropped_result_file}")
                result_df.to_csv(uncropped_result_file)

                # now try to generate the cropped result file 
                generate_cropped_csv_file(
                    input_csv_file=uncropped_result_file,
                    input_bbox_file=bbox_file,
                    output_csv_file=result_file,
                    img_height=320,
                    img_width=320,
                    mode="subtract"  # For mapping back from global to cropped space
                )
                print(f"Saved cropped result to {result_file}")

            # when we don't have cropped results
            else: 
                print(f"No bbox file found for view {view}, saving result as-is")
                result_df.to_csv(result_file)

        # clean up temporary files
        # Clean up temporary files
        for view in views:
            for seed_idx in range(len(seed_dirs)):
                for file_type in ['temp', 'temp_bbox']:
                    temp_file = os.path.join(output_dir, f"{file_type}_{view}_{seed_idx}_{sequence_key}.csv")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

            # # Save uncropped result
            # uncropped_result_file = os.path.join(view_output_path, "concatenated_sequence_uncropped.csv")
            # result_df.to_csv(uncropped_result_file)
            # print("saving uncropped result to ", uncropped_result_file)
            
            # has_bbox_files = False

            # bbox_temp_files = []
            # for seed_idx in csv_files_by_seed.keys():
            #     bbox_temp_file = os.path.join(output_dir, f"temp_bbox_{view}_{seed_idx}_{sequence_key}.csv")
            #     if os.path.exists(bbox_temp_file):
            #         bbox_temp_files.append(bbox_temp_file)
            #         has_bbox_files = True



            # # Final cropped result file path
            # result_file = os.path.join(view_output_path, "concatenated_sequence.csv")
            
            # try:
            #     # Try to create consolidated bbox file
            #     bbox_temp_files = [
            #         os.path.join(output_dir, f"temp_bbox_{view}_{seed_idx}_{sequence_key}.csv")
            #         for seed_idx in csv_files_by_seed.keys()
            #         if os.path.exists(os.path.join(output_dir, f"temp_bbox_{view}_{seed_idx}_{sequence_key}.csv"))
            #     ]
                
            #     if bbox_temp_files:
            #         # Consolidate bbox files
            #         all_bbox_df = pd.DataFrame()
            #         for bbox_file in bbox_temp_files:
            #             try:
            #                 bbox_df = pd.read_csv(bbox_file)
            #                 if not all_bbox_df.empty and 'Unnamed: 0' in all_bbox_df.columns and 'Unnamed: 0' in bbox_df.columns:
            #                     bbox_df['Unnamed: 0'] = bbox_df['Unnamed: 0'] + all_bbox_df['Unnamed: 0'].max() + 1
            #                 all_bbox_df = pd.concat([all_bbox_df, bbox_df]) if not all_bbox_df.empty else bbox_df
            #             except Exception as e:
            #                 print(f"Error reading temp bbox file {bbox_file}: {e}")
                    
            #         if not all_bbox_df.empty:
            #             # Save consolidated bbox file
            #             consolidated_bbox_file = os.path.join(view_output_path, "concatenated_sequence_bbox.csv")
            #             all_bbox_df.to_csv(consolidated_bbox_file, index=False)
                        
            #             # Generate cropped result
            #             generate_cropped_csv_file(
            #                 uncropped_result_file,
            #                 consolidated_bbox_file,
            #                 result_file,
            #                 img_height=320, 
            #                 img_width=320, 
            #                 mode="subtract"
            #             )
            #         else:
            #             result_df.to_csv(result_file)
            #     elif get_bbox_path_fn is not None:
            #         # Try with direct bbox path
            #         bbox_path = get_bbox_path_fn(Path(result_file))
            #         if os.path.exists(bbox_path):
            #             generate_cropped_csv_file(
            #                 uncropped_result_file,
            #                 bbox_path,
            #                 result_file,
            #                 img_height=320,
            #                 img_width=320,
            #                 mode="subtract"
            #             )
            #         else:
            #             result_df.to_csv(result_file)
            #     else:
            #         result_df.to_csv(result_file)
            # except Exception as e:
            #     print(f"Error processing bbox for {view}: {e}")
            #     result_df.to_csv(result_file)
        
        # # Clean up temporary files
        # for view in views:
        #     for seed_idx in range(len(seed_dirs)):
        #         for file_type in ['temp', 'temp_bbox']:
        #             temp_file = os.path.join(output_dir, f"{file_type}_{view}_{seed_idx}_{sequence_key}.csv")
        #             if os.path.exists(temp_file):
        #                 os.remove(temp_file)

# def process_multiview_directories_concat(
#     views: List[str],
#     seed_dirs: List[str],
#     inference_dir: str,
#     output_dir: str,
#     mode: str,
#     pca_object = None,
#     fa_object = None,
#     get_bbox_path_fn: Callable[[Path, ], Path] | None = None,
# ) -> None:
#     """Process all directories for a view by concatenating CSV files per directory,
#     properly handling uncropped versions of files when needed."""
    
#     if fa_object is not None:
#         print("Using FA object for variance inflation")
#     else:
#         print("No FA object available for variance inflation - we are in process_multiview_directories_concat")

#     # Prepare FA parameters for inflation
#     inflate_vars_kwargs = {}
#     if fa_object is not None:
#         try:
#             loading_matrix = fa_object.components_.T
#             mean = fa_object.mean_
            
#             inflate_vars_kwargs = {
#                 'loading_matrix': loading_matrix,
#                 'mean': np.zeros_like(mean)  # To avoid centering twice
#             }
#             print("Successfully extracted FA parameters for variance inflation")
#         except AttributeError as e:
#             print(f"Error extracting FA parameters: {e}. Using default inflation parameters.")
    
#     # Get the original directory structure
#     first_seed_dir = seed_dirs[0]
#     video_dir = os.path.join(first_seed_dir, inference_dir)
    
#     # Find all directories per view and group them
#     directories_by_view = {}
#     for view in views:
#         view_dirs = [d for d in os.listdir(video_dir) if view in d and os.path.isdir(os.path.join(video_dir, d))]
#         directories_by_view[view] = view_dirs
#         print(f"View {view} dirs: {view_dirs}")
    
#     # Group directories across views by their sequence name
#     sequences = {}
#     for view, dirs in directories_by_view.items():
#         for dir_name in dirs:
#             # Extract the sequence name by removing view identifier
#             parts = dir_name.split('_')
#             sequence_key = '_'.join([p for p in parts if p not in views])
            
#             if sequence_key not in sequences:
#                 sequences[sequence_key] = {}
            
#             sequences[sequence_key][view] = dir_name
    
#     # Process each sequence
#     for sequence_key, view_dirs in sequences.items():
#         print(f"\nProcessing sequence: {sequence_key}")
        
#         # For each view in this sequence
#         for view, dir_name in view_dirs.items():
#             print(f"Processing view {view} directory: {dir_name}")
            
#             # Create output directory for this view's sequence
#             view_output_path = os.path.join(output_dir, dir_name)
#             os.makedirs(view_output_path, exist_ok=True)
            
#             # Collect CSV files for each seed
#             all_csv_files = []
#             for seed_dir in seed_dirs:
#                 seed_video_dir = os.path.join(seed_dir, inference_dir)
#                 seed_sequence_dir = os.path.join(seed_video_dir, dir_name)
                
#                 if os.path.exists(seed_sequence_dir):
#                     csv_files = [
#                         f for f in os.listdir(seed_sequence_dir)
#                         if f.endswith(".csv") and not f.endswith("_uncropped.csv")
#                     ]
                    
#                     for csv_file in csv_files:
#                         pred_file = os.path.join(seed_sequence_dir, csv_file)
#                         all_csv_files.append(pred_file)
            
#             # Group CSV files by their ensemble member (seed)
#             csv_files_by_seed = {}
#             for csv_path in all_csv_files:
#                 # Extract seed directory from path
#                 seed_path = csv_path.split(inference_dir)[0].rstrip('/')
#                 if seed_path in seed_dirs:
#                     seed_idx = seed_dirs.index(seed_path)
#                     if seed_idx not in csv_files_by_seed:
#                         csv_files_by_seed[seed_idx] = []
#                     csv_files_by_seed[seed_idx].append(csv_path)
            
#             # Process each ensemble member's CSV files
#             for seed_idx, csv_files in csv_files_by_seed.items():
#                 # Sort CSV files by their frame numbers to ensure proper sequence
#                 csv_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace("img", "")))
                
#                 # Collect bbox files if needed
#                 bbox_files = []
#                 if get_bbox_path_fn is not None:
#                     for csv_file in csv_files:
#                         try:
#                             bbox_path = get_bbox_path_fn(Path(csv_file))
#                             if os.path.exists(bbox_path):
#                                 bbox_files.append(str(bbox_path))
#                         except Exception as e:
#                             print(f"Error finding bbox for {csv_file}: {e}")
                    
#                     # Sort bbox files to match CSV order
#                     if bbox_files:
#                         bbox_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace("img", "").replace("_bbox", "")))
                
#                 # Generate uncropped files if needed
#                 uncropped_files = []
#                 if get_bbox_path_fn is not None and bbox_files:
#                     print(f"Preparing uncropped CSV files for view {view}, seed {seed_idx}")
#                     uncropped_files = prepare_uncropped_csv_files(csv_files, get_bbox_path_fn)
                    
#                     if not uncropped_files:
#                         print(f"Warning: Could not generate uncropped files for view {view}, seed {seed_idx}")
#                         continue
#                 else:
#                     # If no bbox function or files, use original files
#                     uncropped_files = csv_files
                
#                 # Concatenate uncropped files for this seed
#                 concat_df = pd.DataFrame()
#                 for i, csv_file in enumerate(uncropped_files):
#                     try:
#                         df = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
#                         df = io_utils.fix_empty_first_row(df)
                        
#                         # Ensure unique indices by adding offset based on file position
#                         # This prevents duplicate index issues later
#                         if not concat_df.empty:
#                             # Get the last index in the current concat_df
#                             last_idx = concat_df.index[-1]
#                             # Shift indices in the new dataframe
#                             df.index = df.index + last_idx + 1
                        
#                         if concat_df.empty:
#                             concat_df = df
#                         else:
#                             concat_df = pd.concat([concat_df, df])
#                     except Exception as e:
#                         print(f"Error reading {csv_file}: {e}")
                
#                 if not concat_df.empty:
#                     # Save concatenated uncropped file
#                     temp_file = os.path.join(output_dir, f"temp_{view}_{seed_idx}_{sequence_key}.csv")
#                     concat_df.to_csv(temp_file)
#                     print(f"Saved concatenated uncropped file for view {view}, seed {seed_idx}: {temp_file}")
                    
#                     # Update for this seed
#                     csv_files_by_seed[seed_idx] = [temp_file]
                
#                 # If we have bbox files, concatenate them too
#                 if bbox_files:
#                     concat_bbox_df = pd.DataFrame()
#                     for i, bbox_file in enumerate(bbox_files):
#                         try:
#                             bbox_df = pd.read_csv(bbox_file)
                            
#                             # Ensure unique indices by adding offset based on file position
#                             if not concat_bbox_df.empty:
#                                 # Find the maximum index in the current concatenated bbox dataframe
#                                 if 'Unnamed: 0' in concat_bbox_df.columns:
#                                     max_idx = concat_bbox_df['Unnamed: 0'].max()
#                                     # Adjust indices in the new bbox dataframe
#                                     if 'Unnamed: 0' in bbox_df.columns:
#                                         bbox_df['Unnamed: 0'] = bbox_df['Unnamed: 0'] + max_idx + 1
                                
#                                 # Add offset to the frame column if it exists
#                                 if 'frame' in bbox_df.columns and 'frame' in concat_bbox_df.columns:
#                                     max_frame = concat_bbox_df['frame'].max()
#                                     bbox_df['frame'] = bbox_df['frame'] + max_frame + 1
                            
#                             if concat_bbox_df.empty:
#                                 concat_bbox_df = bbox_df
#                             else:
#                                 # Use regular concat to maintain adjusted indices
#                                 concat_bbox_df = pd.concat([concat_bbox_df, bbox_df])
#                         except Exception as e:
#                             print(f"Error reading bbox file {bbox_file}: {e}")
                    
#                     if not concat_bbox_df.empty:
#                         # Save concatenated bbox file for this seed
#                         bbox_temp_file = os.path.join(output_dir, f"temp_bbox_{view}_{seed_idx}_{sequence_key}.csv")
#                         concat_bbox_df.to_csv(bbox_temp_file, index=False)
#                         print(f"Saved temporary concatenated bbox file for view {view}, seed {seed_idx}")
        
#         # Collect all concatenated files for all views
#         all_view_files = {}
        
#         for view in views:
#             if view in view_dirs:
#                 view_files = []
#                 for seed_idx in csv_files_by_seed.keys():
#                     temp_file = os.path.join(output_dir, f"temp_{view}_{seed_idx}_{sequence_key}.csv")
#                     if os.path.exists(temp_file):
#                         view_files.append(temp_file)
#                 if view_files:
#                     all_view_files[view] = view_files
        
#         # Run EKS multiview on the concatenated files
#         input_dfs_list = []
#         keypoint_names = []
        
#         for view, files in all_view_files.items():
#             if files:
#                 view_dfs, kp_names = format_data(
#                     input_source=files,
#                     camera_names=[view]
#                 )
#                 if not keypoint_names:
#                     keypoint_names = kp_names
#                 input_dfs_list.extend(view_dfs)
        
#         if input_dfs_list and keypoint_names:
#             print(f"Running EKS multiview on concatenated sequence {sequence_key}")
#             print(f"Using {len(input_dfs_list)} input DataFrames")
            
#             print(f"the inflation vars kwargs are {inflate_vars_kwargs}")
#             print(f" the pca object is {pca_object}")

#             print(f" input_dfs_list is {input_dfs_list}")
            
#             # Run EKS multiview on concatenated data (these are already uncropped)
#             results_dfs = run_eks_multiview(
#                 markers_list=input_dfs_list,
#                 keypoint_names=keypoint_names,
#                 views=views,
#                 quantile_keep_pca=50,
#                 pca_object=pca_object,
#                 inflate_vars_kwargs=inflate_vars_kwargs
#             )
            
#             # Save results for each view
#             for view, result_df in results_dfs.items():
#                 if view in view_dirs:
#                     # Get the view-specific directory name
#                     view_dir = view_dirs[view]
#                     # Create path for view results
#                     view_output_path = os.path.join(output_dir, view_dir)
#                     os.makedirs(view_output_path, exist_ok=True)
                    
#                     # Save the uncropped result first
#                     uncropped_result_file = os.path.join(view_output_path, "concatenated_sequence_uncropped.csv")
#                     result_df.to_csv(uncropped_result_file)
#                     print(f"Saved uncropped EKS results for view {view} to {uncropped_result_file}")
                    
#                     # Create the final cropped result file path
#                     result_file = os.path.join(view_output_path, "concatenated_sequence.csv")
                    
#                     # Try to generate a consolidated bbox file for the entire sequence
#                     try:
#                         # Collect all temporary bbox files for this view
#                         bbox_temp_files = []
#                         for seed_idx in csv_files_by_seed.keys():
#                             bbox_temp_file = os.path.join(output_dir, f"temp_bbox_{view}_{seed_idx}_{sequence_key}.csv")
#                             if os.path.exists(bbox_temp_file):
#                                 bbox_temp_files.append(bbox_temp_file)
                        
#                         # If we have temporary bbox files, consolidate them
#                         if bbox_temp_files:
#                             print(f"Consolidating bbox files for view {view}")
#                             all_bbox_df = pd.DataFrame()
                            
#                             for bbox_file in bbox_temp_files:
#                                 try:
#                                     bbox_df = pd.read_csv(bbox_file)
#                                     if all_bbox_df.empty:
#                                         all_bbox_df = bbox_df
#                                     else:
#                                         # Find the maximum index in the current bbox dataframe
#                                         if 'Unnamed: 0' in all_bbox_df.columns and 'Unnamed: 0' in bbox_df.columns:
#                                             max_idx = all_bbox_df['Unnamed: 0'].max()
#                                             # Adjust indices in the new bbox dataframe
#                                             bbox_df['Unnamed: 0'] = bbox_df['Unnamed: 0'] + max_idx + 1
                                        
#                                         # Use regular concat to maintain adjusted indices
#                                         all_bbox_df = pd.concat([all_bbox_df, bbox_df])
#                                 except Exception as e:
#                                     print(f"Error reading temp bbox file {bbox_file}: {e}")
                            
#                             if not all_bbox_df.empty:
#                                 # Save consolidated bbox file
#                                 consolidated_bbox_file = os.path.join(view_output_path, "concatenated_sequence_bbox.csv")
#                                 all_bbox_df.to_csv(consolidated_bbox_file, index=False)
#                                 print(f"Saved consolidated bbox file for view {view}")
                                
#                                 # Now use this bbox file to generate the cropped result
#                                 try:
#                                     print(f"Generating cropped CSV file for {view}")
#                                     # Process with cropping to original space
#                                     generate_cropped_csv_file(
#                                         uncropped_result_file,  # uncropped result file
#                                         consolidated_bbox_file,  # consolidated bbox file
#                                         result_file,  # output cropped file
#                                         img_height=320, 
#                                         img_width=320, 
#                                         mode="subtract"  # subtract bbox coordinates to get back to cropped space
#                                     )
#                                     print(f"Successfully saved cropped EKS results for view {view} to {result_file}")
#                                 except Exception as e:
#                                     print(f"Error during cropping with consolidated bbox: {e}")
#                                     # Fall back to original file if cropping fails
#                                     result_df.to_csv(result_file)
#                                     print(f"WARNING: Saved uncropped results as main output due to cropping error")
#                         else:
#                             # No bbox files, try with get_bbox_path_fn
#                             print(f"No consolidated bbox files found for {view}, trying with direct bbox path")
#                             try:
#                                 if get_bbox_path_fn is not None:
#                                     bbox_path = get_bbox_path_fn(Path(result_file))
#                                     if os.path.exists(bbox_path):
#                                         generate_cropped_csv_file(
#                                             uncropped_result_file,
#                                             bbox_path,
#                                             result_file,
#                                             img_height=320,
#                                             img_width=320,
#                                             mode="subtract"
#                                         )
#                                         print(f"Successfully saved cropped EKS results using direct bbox path for {view}")
#                                     else:
#                                         print(f"No bbox file found at {bbox_path}")
#                                         result_df.to_csv(result_file)
#                                         print(f"WARNING: Saved uncropped results as main output due to missing bbox file")
#                                 else:
#                                     result_df.to_csv(result_file)
#                                     print(f"WARNING: No bbox function provided, saving uncropped results as main output")
#                             except Exception as e:
#                                 print(f"Error with direct bbox path: {e}")
#                                 result_df.to_csv(result_file)
#                                 print(f"WARNING: Saved uncropped results as main output due to error")
#                     except Exception as e:
#                         print(f"Error processing bbox files for {view}: {e}")
#                         # If all else fails, at least save the uncropped result as the main output
#                         result_df.to_csv(result_file)
#                         print(f"WARNING: Saved uncropped results as main output due to processing error")
        
#         # Clean up temporary files
#         for view in views:
#             for seed_idx in range(len(seed_dirs)):
#                 # Remove temp concatenated files
#                 temp_file = os.path.join(output_dir, f"temp_{view}_{seed_idx}_{sequence_key}.csv")
#                 if os.path.exists(temp_file):
#                     os.remove(temp_file)
                
#                 # Remove temp bbox files
#                 bbox_temp_file = os.path.join(output_dir, f"temp_bbox_{view}_{seed_idx}_{sequence_key}.csv")
#                 if os.path.exists(bbox_temp_file):
#                     os.remove(bbox_temp_file)
            
#             print(f"Cleaned up temporary files for view {view}")

# def process_multiview_directories_concat(
#     views: List[str],
#     seed_dirs: List[str],
#     inference_dir: str,
#     output_dir: str,
#     mode: str,
#     pca_object = None,
#     fa_object = None,
#     get_bbox_path_fn: Callable[[Path, ], Path] | None = None,
# ) -> None:
#     """Process all directories for a view by concatenating CSV files per directory,
#     properly handling uncropped versions of files when needed."""
    
#     if fa_object is not None:
#         print("Using FA object for variance inflation")
#     else:
#         print("No FA object available for variance inflation - we are in process_multiview_directories_concat")

#     # Prepare FA parameters for inflation
#     inflate_vars_kwargs = {}
#     if fa_object is not None:
#         try:
#             loading_matrix = fa_object.components_.T
#             mean = fa_object.mean_
            
#             inflate_vars_kwargs = {
#                 'loading_matrix': loading_matrix,
#                 'mean': np.zeros_like(mean)  # To avoid centering twice
#             }
#             print("Successfully extracted FA parameters for variance inflation")
#         except AttributeError as e:
#             print(f"Error extracting FA parameters: {e}. Using default inflation parameters.")
    
#     # Get the original directory structure
#     first_seed_dir = seed_dirs[0]
#     video_dir = os.path.join(first_seed_dir, inference_dir)
    
#     # Find all directories per view and group them
#     directories_by_view = {}
#     for view in views:
#         view_dirs = [d for d in os.listdir(video_dir) if view in d and os.path.isdir(os.path.join(video_dir, d))]
#         directories_by_view[view] = view_dirs
#         print(f"View {view} dirs: {view_dirs}")
    
#     # Group directories across views by their sequence name
#     sequences = {}
#     for view, dirs in directories_by_view.items():
#         for dir_name in dirs:
#             # Extract the sequence name by removing view identifier
#             parts = dir_name.split('_')
#             sequence_key = '_'.join([p for p in parts if p not in views])
            
#             if sequence_key not in sequences:
#                 sequences[sequence_key] = {}
            
#             sequences[sequence_key][view] = dir_name
    
#     # Process each sequence
#     for sequence_key, view_dirs in sequences.items():
#         print(f"\nProcessing sequence: {sequence_key}")
        
#         # For each view in this sequence
#         for view, dir_name in view_dirs.items():
#             print(f"Processing view {view} directory: {dir_name}")
            
#             # Create output directory for this view's sequence
#             view_output_path = os.path.join(output_dir, dir_name)
#             os.makedirs(view_output_path, exist_ok=True)
            
#             # Collect CSV files for each seed
#             all_csv_files = []
#             for seed_dir in seed_dirs:
#                 seed_video_dir = os.path.join(seed_dir, inference_dir)
#                 seed_sequence_dir = os.path.join(seed_video_dir, dir_name)
                
#                 if os.path.exists(seed_sequence_dir):
#                     csv_files = [
#                         f for f in os.listdir(seed_sequence_dir)
#                         if f.endswith(".csv") and not f.endswith("_uncropped.csv")
#                     ]
                    
#                     for csv_file in csv_files:
#                         pred_file = os.path.join(seed_sequence_dir, csv_file)
#                         all_csv_files.append(pred_file)
            
#             # Group CSV files by their ensemble member (seed)
#             csv_files_by_seed = {}
#             for csv_path in all_csv_files:
#                 seed_idx = seed_dirs.index(csv_path.split(inference_dir)[0].rstrip('/'))
#                 if seed_idx not in csv_files_by_seed:
#                     csv_files_by_seed[seed_idx] = []
#                 csv_files_by_seed[seed_idx].append(csv_path)
            
#             # Process each ensemble member's CSV files
#             uncropped_files_by_seed = {}
#             for seed_idx, csv_files in csv_files_by_seed.items():
#                 # Sort CSV files by their frame numbers to ensure proper sequence
#                 csv_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace("img", "")))
                
#                 # Check if we need to prepare uncropped versions
#                 if get_bbox_path_fn is not None:
#                     uncropped_files = prepare_uncropped_csv_files(csv_files, get_bbox_path_fn)
#                     if uncropped_files:
#                         uncropped_files_by_seed[seed_idx] = uncropped_files
#                         # Use uncropped files for concatenation if they exist
#                         concat_source_files = uncropped_files
#                         print(f"Using uncropped files for concatenation for view {view}, seed {seed_idx}")
#                     else:
#                         # If no uncropped files, use original files
#                         concat_source_files = csv_files
#                 else:
#                     # No bbox function, use original files
#                     concat_source_files = csv_files
                
#                 # Concatenate CSV files for this ensemble member
#                 concat_dfs = []
#                 for csv_file in concat_source_files:
#                     try:
#                         df = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
#                         df = io_utils.fix_empty_first_row(df)
#                         concat_dfs.append(df)
#                     except Exception as e:
#                         print(f"Error reading {csv_file}: {e}")
                
#                 if concat_dfs:
#                     # Concatenate all DataFrames
#                     concat_df = pd.concat(concat_dfs)
                    
#                     # Save concatenated file temporarily
#                     temp_file = os.path.join(output_dir, f"temp_{view}_{seed_idx}_{sequence_key}.csv")
#                     concat_df.to_csv(temp_file)
#                     print(f"Saved concatenated file for view {view}, seed {seed_idx}: {temp_file}")
                    
#                     # Update all_csv_files for this seed with the concatenated file
#                     csv_files_by_seed[seed_idx] = [temp_file]
        
#         # Collect all concatenated files for all views
#         all_view_files = {}
#         for view in views:
#             if view in view_dirs:
#                 view_files = []
#                 for seed_idx in csv_files_by_seed.keys():
#                     temp_file = os.path.join(output_dir, f"temp_{view}_{seed_idx}_{sequence_key}.csv")
#                     if os.path.exists(temp_file):
#                         view_files.append(temp_file)
#                 all_view_files[view] = view_files
        
#         # Run EKS multiview on the concatenated files
#         input_dfs_list = []
#         keypoint_names = []
        
#         for view, files in all_view_files.items():
#             if files:
#                 view_dfs, kp_names = format_data(
#                     input_source=files,
#                     camera_names=[view]
#                 )
#                 if not keypoint_names:
#                     keypoint_names = kp_names
#                 input_dfs_list.extend(view_dfs)
        
#         if input_dfs_list and keypoint_names:
#             print(f"Running EKS multiview on concatenated sequence {sequence_key}")
#             print(f"Using {len(input_dfs_list)} input DataFrames")
            
#             print(f"the inflation vars kwargs are {inflate_vars_kwargs}")
#             print(f" the pca object is {pca_object}")
#             # Run EKS multiview on concatenated data

#             print (f" input_dfs_list is {input_dfs_list}")
            
#             results_dfs = run_eks_multiview(
#                 markers_list=input_dfs_list,
#                 keypoint_names=keypoint_names,
#                 views=views,
#                 quantile_keep_pca=50,
#                 pca_object=pca_object,
#                 inflate_vars_kwargs=inflate_vars_kwargs
#             )
            
#             # Save results for each view
#             for view, result_df in results_dfs.items():
#                 if view in view_dirs:
#                     # Get the view-specific directory name
#                     view_dir = view_dirs[view]
#                     # Create path for view results
#                     view_output_path = os.path.join(output_dir, view_dir)
#                     os.makedirs(view_output_path, exist_ok=True)

#                     # Define result file paths
#                     result_file = Path(os.path.join(view_output_path, "concatenated_sequence.csv"))
                    
#                     # Determine if we need uncropped result
#                     has_uncropped = any(uncropped_files_by_seed for seed_idx, uncropped_files_by_seed in uncropped_files_by_seed.items())
                    
#                     if has_uncropped:
#                         # Save uncropped result first
#                         uncropped_result_file = result_file.with_stem(result_file.stem + "_uncropped")
#                         result_df.to_csv(uncropped_result_file)
#                         print(f"Saved uncropped EKS results for view {view} to {uncropped_result_file}")
#                         # we need to concatenate the bbox files so that we can generate the cropped version

                        
#                         # Generate cropped version
#                         try:
#                             bbox_path = get_bbox_path_fn(Path(result_file))
#                             generate_cropped_csv_file(
#                                 uncropped_result_file, 
#                                 bbox_path, 
#                                 result_file, 
#                                 img_height=320, 
#                                 img_width=320, 
#                                 mode="subtract"
#                             )
#                             print(f"Saved cropped EKS results for view {view} to {result_file}")
#                         except Exception as e:
#                             print(f"Error generating cropped file for {view}: {e}")
#                             # Fall back to saving original results
#                             result_df.to_csv(result_file)
#                     else:
#                         # Just save original results
#                         result_df.to_csv(result_file)
#                         print(f"Saved EKS results for view {view} to {result_file}")
        
#         # Clean up temporary files
#         for view in views:
#             for seed_idx in range(len(seed_dirs)):
#                 temp_file = os.path.join(output_dir, f"temp_{view}_{seed_idx}_{sequence_key}.csv")
#                 if os.path.exists(temp_file):
#                     os.remove(temp_file)
#                     print(f"Removed temporary file: {temp_file}")


# def process_singleview_directory(
#     original_dir: str,
#     csv_files: List[str],
#     view: str,
#     seed_dirs: List[str],
#     inference_dir: str,
#     output_dir: str,
#     mode: str
# ) -> None:
#     """Process a single view directory"""
#     print(f"Processing directory: {original_dir}")
    
#     # Identify current view from directory name
#     curr_view = next((part for part in original_dir.split("_") if part in [view]), 
#                      original_dir.split("_")[-1])
    
#     print(f"Identified view: {curr_view}")
    
#     # Create output directory matching original structure
#     sequence_output_dir = os.path.join(output_dir, original_dir)
#     os.makedirs(sequence_output_dir, exist_ok=True)
    
#     # Process each CSV file in the directory
#     for csv_file in csv_files:
#         print(f"Processing file: {csv_file}")
        
#         all_pred_files = []
        
#         # Get directory name for current view
#         curr_dir_name = original_dir.replace(curr_view, view) if curr_view in original_dir else original_dir
        
#         for seed_dir in seed_dirs:
#             seed_video_dir = os.path.join(seed_dir, inference_dir)
#             seed_sequence_dir = os.path.join(seed_video_dir, curr_dir_name)
            
#             if os.path.exists(seed_sequence_dir):
#                 pred_file = os.path.join(seed_sequence_dir, csv_file)
#                 if os.path.exists(pred_file):
#                     all_pred_files.append(pred_file)
        
#         if all_pred_files:
#             # Process ensemble data
#             input_dfs_list, keypoint_names = format_data(
#                 input_source=all_pred_files,
#                 camera_names=None,
#             )
            
#             # Get column structure and process based on mode
#             if mode in ['ensemble_mean', 'ensemble_median']:
#                 column_structure, _, _, _, _ = process_predictions(all_pred_files[0])
#                 results_df = process_ensemble_frames(all_pred_files, keypoint_names, mode=mode)
#             elif mode == 'eks_singleview':
#                 results_df = run_eks_singleview(
#                     markers_list=input_dfs_list,
#                     keypoint_names=keypoint_names
#                 )
#             else:
#                 print(f"Invalid mode: {mode}")
#                 continue
            
#             # Save results
#             result_file = os.path.join(sequence_output_dir, csv_file)
#             results_df.to_csv(result_file)
#             print(f"Saved ensemble {mode} predictions to {result_file}")



def process_single_video_view(
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
    
    # Find all CSV files that contain the view name
    view_files = [f for f in base_files if view in f and f.endswith('.csv')]
    
    if not view_files:
        print(f"No files found for view {view}")
        return
        
    # Process each file (representing a different video) separately
    for file_name in view_files:
        print(f"Processing file: {file_name}")
        
        # Collect this file from all seed directories
        pred_files = []
        for seed_dir in seed_dirs:
            pred_file = os.path.join(seed_dir, inference_dir, file_name)
            if os.path.exists(pred_file):
                pred_files.append(pred_file)
        
        if not pred_files:
            print(f"No prediction files found across seeds for {file_name}")
            continue
            
        # Process ensemble data
        input_dfs_list, keypoint_names = format_data(
            input_source=pred_files,
            camera_names=None,
        )
        
        column_structure, _, _, _, _ = process_predictions(pred_files[0])
        
        # Process based on mode
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
    """Post-process ensemble labels by concatenating CSV files per directory"""

    # Load PCA object if it doesn't exist
    if pca_object is None:
        # Check if it exists in the global scope
        global_pca = globals().get('pca_object')
        if global_pca is not None:
            pca_object = global_pca
            print("Loaded PCA object from global scope")
        else:
            # Try to load from the file
            try:
                with open(pca_model_path, "rb") as f:
                    pca_object = CustomUnpickler(f).load()
                print(f"Loaded PCA model from {pca_model_path} and the PCA components shape: {pca_object.components_.shape} ")
            except Exception as e:
                print(f"Could not load PCA model: {e}")

    if fa_object is None:
        global_fa = globals().get('fa_object')
        if fa_object is not None:
            fa_object = global_fa
            print("Loaded FA object from global scope")
            print("Using FA object for variance inflation")
        else:
            # Try to load from the file
            try:
                with open(fa_model_path, "rb") as f:
                    fa_object = CustomUnpickler(f).load()
                print(f"Loaded FA model from {fa_model_path} and the FA components shape: {fa_object.components_.shape} ")
            except Exception as e:
                print(f"Could not load FA model: {e}")
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
                process_final_predictions_concat(
                    view=view,
                    output_dir=output_dir,
                    seed_dirs=seed_dirs,
                    inference_dir=inference_dir,
                    cfg_lp=cfg_lp
                )
        else:
            # Use original implementation for other modes
            print(f"Mode {mode} does not use the concatenation approach. Using original implementation.")

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
                process_single_video_view(
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


