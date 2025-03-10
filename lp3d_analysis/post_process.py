import os
import pandas as pd
import numpy as np
import pickle 

from typing import Optional, Union, Callable

from omegaconf import DictConfig
from typing import List, Literal, Tuple, Dict, Any 
from pathlib import Path

from lightning_pose.utils.cropzoom import generate_cropped_csv_file
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

# pca_model_path = "/teamspace/studios/this_studio/pca_object_inD_fly.pkl"
pca_model_path = "/teamspace/studios/this_studio/pca_object_inD_mirror-mouse-separate.pkl"
# fa_model_path = "/teamspace/studios/this_studio/fa_object_inD_fly.pkl"
fa_model_path = "/teamspace/studios/this_studio/pca_object_inD_mirror-mouse-separate.pkl"

# Custom function to force pickle to find NaNPCA2 in pca_global
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "NaNPCA":
            from lightning_pose.utils.pca import  NaNPCA
            return NaNPCA
        elif name == "EnhancedFactorAnalysis":
            from lp3d_analysis.pca_global import EnhancedFactorAnalysis
            return EnhancedFactorAnalysis
        return super().find_class(module, name)

# Load PCA object before defining functions
try:
    with open(pca_model_path, "rb") as f:
        pca_object = CustomUnpickler(f).load()
    print(f"PCA model loaded successfully from {pca_model_path}.")
except AttributeError as e:
    print(f"Error loading PCA model: {e}. Ensure NaNPCA2 is correctly imported from pca_global.py.")
except FileNotFoundError as e:
    print(f"Skipping loading pca_object from {pca_model_path}:")
    print(e)

try:
    with open(fa_model_path, "rb") as f:
        fa_object = CustomUnpickler(f).load()
    print(f"PCA model loaded successfully from {fa_model_path}. for the purpose of variance inflation.")
except AttributeError as e:
    print(f"Error loading PCA model: {e}. Ensure NaNPCA2 is correctly imported from pca_global.py.")
except FileNotFoundError as e:
    print(f"Skipping loading pca_object from {pca_model_path}:")
    print(e)

# # Load FA object before defining functions
# try:
#     with open(fa_model_path, "rb") as f:
#         fa_object = CustomUnpickler(f).load()
#     print(f"FA model loaded successfully from {fa_model_path}.")
# except AttributeError as e:
#     print(f"Error loading FA model: {e}. Ensure EnhancedFactorAnalysis is correctly imported from pca_global.py.")



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


def process_final_predictions(
    view: str,
    output_dir: str,
    seed_dirs: List[str],
    inference_dir: str,
    cfg_lp: DictConfig,
) -> None:
    """Process and save final predictions for a view, computing metrics"""
    
    # Load original CSV file to get frame paths
    orig_pred_file = os.path.join(seed_dirs[0], inference_dir, f'predictions_{view}_new.csv')
    original_df = pd.read_csv(orig_pred_file, header=[0, 1, 2], index_col=0)
    original_index = original_df.index
    results_list = []
    
    # Process each path from original predictions
    for img_path in original_index:
        # Get sequence name and base filename
        sequence_name = img_path.split('/')[1]  # e.g., "180623_000_bot"
        base_filename = os.path.splitext(os.path.basename(img_path))[0]  # e.g., "img107262"
        
        # Get the sequence CSV path
        snippet_file = os.path.join(output_dir, sequence_name, f"{base_filename}.csv")
        print(f"\nProcessing {img_path}")
        print(f"Looking for file: {snippet_file}")
        
        if os.path.exists(snippet_file):
            try:
                # Load the CSV with multi-index columns
                snippet_df = pd.read_csv(snippet_file, header=[0, 1, 2], index_col=0)
                print(f"Loaded snippet file with shape: {snippet_df.shape}")
                
                # Ensure odd number of frames
                assert snippet_df.shape[0] & 1 == 1, f"Expected odd number of frames, got {snippet_df.shape[0]}"
                
                # Get center frame index
                idx_frame = int(np.floor(snippet_df.shape[0] / 2))
                print(f"Extracting center frame at index {idx_frame}")
                
                # Extract center frame and set index
                center_frame = snippet_df.iloc[[idx_frame]]  # Keep as DataFrame with one row
                center_frame.index = [img_path]
                results_list.append(center_frame)
                
                print(f"Successfully extracted center frame")
                
            except Exception as e:
                print(f"Error processing {snippet_file}: {str(e)}")
                print("Full error:")
                import traceback
                traceback.print_exc()
        else:
            print(f"Warning: Could not find file {snippet_file}")
    
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
            compute_metrics(cfg=cfg_lp_view, preds_file=preds_file, data_module=None)
            print(f"Successfully computed metrics for {preds_file}")
        except Exception as e:
            print(f"Error computing metrics for {view}: {str(e)}")
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
            csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
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


# Modified process_multiview_directory function to handle results for all views properly
def process_multiview_directory(
    original_dir: str,
    csv_files: List[str],
    curr_view: str,
    views: List[str],
    seed_dirs: List[str],
    inference_dir: str,
    output_dir: str,
    mode: str,
    pca_object = None,
    fa_object = None,
    get_bbox_path_fn: Callable[[Path, ], Path] | None = None,
) -> None:
    """Process a multiview directory for EKS multiview mode"""

    # Get the global FA object if none is provided
    if fa_object is None:
        fa_object = globals().get('fa_object')
        if fa_object is not None:
            print("Using global FA object for multiview processing")
        else:
            print("Warning: No FA object available for variance inflation")

    # if fa_object is not None:
    #     print("Using FA object for variance inflation")
    # else:
    #     print("No FA object available for variance inflation - we are in process_multiview_directory")

    # Create output directories for ALL views in advance
    for view in views:
        # Replace curr_view with each view to get view-specific directory names
        view_dir_name = original_dir.replace(curr_view, view)
        view_output_dir = os.path.join(output_dir, view_dir_name)
        os.makedirs(view_output_dir, exist_ok=True)
        print(f"Created output directory for {view}: {view_output_dir}")

    # Prepare FA parameters for inflation
    inflate_vars_kwargs = {}
    if fa_object is not None:
        # Extract loading matrix and mean from the FA object
        try:
            loading_matrix = fa_object.components_.T  # Typically the loading matrix is the transpose of components_
            mean = fa_object.mean_
            
            inflate_vars_kwargs = {
                'loading_matrix': loading_matrix,
                'mean': mean,
                # 'mean': np.zeros_like(mean)  # we had an issue of centering twice 
            }
            print("Successfully extracted FA parameters for variance inflation")
        except AttributeError as e:
            print(f"Error extracting FA parameters: {e}. Using default inflation parameters.")

    # Process each CSV file only once
    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")

        all_pred_files = []
        # Collect prediction files for all views
        for view in views:
            # Create view-specific directory name
            view_dir_name = original_dir.replace(curr_view, view)

            for seed_dir in seed_dirs:
                seed_video_dir = os.path.join(seed_dir, inference_dir)
                seed_sequence_dir = os.path.join(seed_video_dir, view_dir_name)

                if os.path.exists(seed_sequence_dir):
                    pred_file = os.path.join(seed_sequence_dir, csv_file)
                    if os.path.exists(pred_file):
                        all_pred_files.append(pred_file)

        if all_pred_files:

            def _prepare_multiview_eks_uncropped_predictions(all_pred_files: list[str]):
                # Check if a bbox file exists
                # has_checked_bbox_file caches the file check
                has_checked_bbox_file = False
                all_pred_files_uncropped = []
                for p in all_pred_files:
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
                    all_pred_files_uncropped.append(str(remapped_p))
                    generate_cropped_csv_file(p, bbox_path, remapped_p, mode="add")

                return all_pred_files_uncropped

            # Not None when there's bbox files in the dataset, i.e. chickadee-crop.
            all_pred_files_uncropped = (
                _prepare_multiview_eks_uncropped_predictions(all_pred_files)
                if get_bbox_path_fn is not None
                else None
            )
            # Get input data and run EKS - only once per CSV file, not per view
            input_dfs_list, keypoint_names = format_data(
                input_source=all_pred_files_uncropped or all_pred_files,
                camera_names=views,
            )

            print(f"the inflation vars kwargs are {inflate_vars_kwargs}")
            print(f" the pca object is {pca_object}")
            results_dfs = run_eks_multiview(
                markers_list=input_dfs_list,
                keypoint_names=keypoint_names,
                views=views,
                quantile_keep_pca=100, # this is for the labeled frames
                pca_object=pca_object,
                inflate_vars_kwargs=inflate_vars_kwargs
            )
            
            # Save results for each view
            for view, result_df in results_dfs.items():
                # Create the view-specific directory name and output path
                view_dir_name = original_dir.replace(curr_view, view)
                view_output_dir = Path(output_dir) / view_dir_name
                
                # Save to the view-specific directory
                result_file = view_output_dir / csv_file
                # If cropped dataset, save to a _uncropped file, so that we can later undo the remapping.
                if all_pred_files_uncropped is not None:
                    uncropped_result_file = result_file.with_stem(result_file.stem + "_uncropped")
                else:
                    uncropped_result_file = None

                result_df.to_csv(uncropped_result_file or result_file)

                # Crop the multiview-eks output back to original cropped coordinate space.
                if all_pred_files_uncropped is not None:
                    bbox_path = get_bbox_path_fn(result_file)
                    generate_cropped_csv_file(uncropped_result_file, bbox_path, result_file, mode="subtract")

                print(f"Saved EKS results for view {view} to {result_file}")


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
    
    # Create output directory matching original structure
    sequence_output_dir = os.path.join(output_dir, original_dir)
    os.makedirs(sequence_output_dir, exist_ok=True)
    
    # Process each CSV file in the directory
    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        
        all_pred_files = []
        
        # Get directory name for current view
        curr_dir_name = original_dir.replace(curr_view, view) if curr_view in original_dir else original_dir
        
        for seed_dir in seed_dirs:
            seed_video_dir = os.path.join(seed_dir, inference_dir)
            seed_sequence_dir = os.path.join(seed_video_dir, curr_dir_name)
            
            if os.path.exists(seed_sequence_dir):
                pred_file = os.path.join(seed_sequence_dir, csv_file)
                if os.path.exists(pred_file):
                    all_pred_files.append(pred_file)
        
        if all_pred_files:
            # Process ensemble data
            input_dfs_list, keypoint_names = format_data(
                input_source=all_pred_files,
                camera_names=None,
            )
            
            # Get column structure and process based on mode
            if mode in ['ensemble_mean', 'ensemble_median']:
                column_structure, _, _, _, _ = process_predictions(all_pred_files[0])
                results_df = process_ensemble_frames(all_pred_files, keypoint_names, mode=mode)
            elif mode == 'eks_singleview':
                results_df = run_eks_singleview(
                    markers_list=input_dfs_list,
                    keypoint_names=keypoint_names
                )
            else:
                print(f"Invalid mode: {mode}")
                continue
            
            # Save results
            result_file = os.path.join(sequence_output_dir, csv_file)
            results_df.to_csv(result_file)
            print(f"Saved ensemble {mode} predictions to {result_file}")

def process_single_video_view(
    view: str,
    seed_dirs: List[str],
    inference_dir: str,
    output_dir: str,
    mode: str
) -> None:
    """Process a single video view"""
    stacked_arrays = []
    stacked_dfs = []
    column_structure = None
    keypoint_names = []
    df_index = None
    pred_files = []
    
    # Collect prediction files for this view across all seeds
    for seed_dir in seed_dirs:
        base_files = os.listdir(os.path.join(seed_dir, inference_dir))
        csv_files = [f for f in base_files if view in f]
        
        # Assert that there's only one matching file
        assert len(csv_files) == 1, f"Expected 1 file for view {view}, found {len(csv_files)}"
        
        pred_file = os.path.join(seed_dir, inference_dir, csv_files[0])
        pred_files.append(pred_file)
    
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
        return
    
    # Save results using the same filename pattern
    base_files = os.listdir(os.path.join(seed_dirs[0], inference_dir))
    csv_files = [f for f in base_files if view in f]
    
    if csv_files:
        base_name = csv_files[0]
        preds_file = os.path.join(output_dir, base_name)
        results_df.to_csv(preds_file)
        print(f"Saved ensemble {mode} predictions for {view} view to {preds_file}")




def post_process_ensemble_labels(
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
    """Post-process ensemble labels"""

    # Use the global pca_object if none is provided explicitly
    if pca_object is None:
        # Use the global pca_object defined at the module level
        pca_object = globals().get('pca_object')
        if pca_object is not None:
            print("Using global PCA object for post-processing")
        else:
            print("No PCA object available for post-processing")
    
    # Use the global fa_object if none is provided explicitly
    if fa_object is None:
        fa_object = globals().get('fa_object')
        if fa_object is not None:
            print("Using global FA object for post-processing")
        else:
            print("No FA object available for post-processing")

    # if fa_object is not None:
    #     print("Using FA object for variance inflation")
    # else:
    #     print("No FA object available for variance inflation we are in post_process_ensemble_labels")

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
            continue  # Skip this inference directory
            
        print(f"Subdirectories found in {inf_dir_in_first_seed}. Processing {inference_dir}...")
        output_dir = os.path.join(ensemble_dir, mode, inference_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get original directory structure
        original_structure = get_original_structure(first_seed_dir, inference_dir, views)
        
        if mode == 'eks_multiview':
            # For multiview mode, process each sequence once, not per view
            # Group directories by sequence (removing the view-specific part)
            sequences = {}
            for original_dir in original_structure.keys():
                # Extract the sequence name by removing view identifier
                curr_view = next((part for part in original_dir.split("_") if part in views), 
                               original_dir.split("_")[-1])
                
                # Create a sequence key by replacing the view with a placeholder
                sequence_key = original_dir.replace(curr_view, "VIEW_PLACEHOLDER")
                
                if sequence_key not in sequences:
                    sequences[sequence_key] = []
                
                # Store the original directory and its view
                sequences[sequence_key].append((original_dir, curr_view))
            
            # Process each sequence once (not per view)
            for sequence_key, dir_view_pairs in sequences.items():
                # Take the first directory as reference for processing
                first_dir, first_view = dir_view_pairs[0]
                csv_files = original_structure[first_dir]
                
                print(f"Processing sequence: {sequence_key} with {len(csv_files)} CSV files")

                def _get_bbox_path(p: Path) -> Path:
                    """Given some preds file p, it will return the bbox path."""
                    n_levels_up = len(p.parent.parts) - len(Path(results_dir).parts)
                    if "eks_multiview" in p.parts:
                        n_levels_up -= 1
                    p_minus_model_dir = Path(p).relative_to(p.parents[n_levels_up])
                    p_rooted_at_data_dir = Path(cfg_lp.data.data_dir) / p_minus_model_dir
                    bbox_path = p_rooted_at_data_dir.with_stem(p.stem + "_bbox")
                    return bbox_path

                # Process this sequence only once
                process_multiview_directory(
                    first_dir, csv_files, first_view, views, 
                    seed_dirs, inference_dir, output_dir, mode,
                    pca_object=pca_object,
                    fa_object=fa_object,
                    get_bbox_path_fn=_get_bbox_path
                )
            
            # Process final predictions for all views
            for view in views:
                process_final_predictions(
                    view=view,
                    output_dir=output_dir,
                    seed_dirs=seed_dirs,
                    inference_dir=inference_dir,
                    cfg_lp=cfg_lp
                )
                
        else:
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
                    cfg_lp=cfg_lp
                )


def post_process_ensemble_videos(
    cfg_lp: DictConfig,
    results_dir: str,
    model_type: str,
    n_labels: int,
    seed_range: tuple[int, int],
    views: list[str], 
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
        print(f"Post-processing ensemble predictions for {model_type} {n_labels} {seed_range[0]}-{seed_range[1]} for {inference_dirs}")
        first_seed_dir = seed_dirs[0]
        inf_dir_in_first_seed = os.path.join(first_seed_dir, inference_dir)
        
        # Check if the inference_dir contains ANY subdirectories
        entries = os.listdir(inf_dir_in_first_seed)
        contains_subdirectory = any(os.path.isdir(os.path.join(inf_dir_in_first_seed, entry)) for entry in entries)
        
        if contains_subdirectory:
            print(f"Found subdirectories in {inf_dir_in_first_seed}. Skipping.")
            continue  # Skip if there are any subdirectories
        
        print(f"Directory contains only files. Processing {inference_dir}...")
        output_dir = os.path.join(ensemble_dir, mode, inference_dir)
        os.makedirs(output_dir, exist_ok=True)

        if mode == 'eks_multiview':
            # Process multiview case for videos
            input_dfs_list = []
            view_indices = {}
            all_pred_files = []

            # Collect files for all views and seeds
            for view in views:
                print(f"Processing view: {view}")
                pred_files_for_view = []
                
                for seed_dir in seed_dirs:
                    base_files = os.listdir(os.path.join(seed_dir, inference_dir))
                    pred_files = [
                        os.path.join(seed_dir, inference_dir, f)
                        for f in base_files 
                        if view in f 
                    ]
                    pred_files_for_view.extend(pred_files)

                if pred_files_for_view:
                    # Get sample dataframe to extract indices
                    sample_df = pd.read_csv(pred_files_for_view[0], header=[0, 1, 2], index_col=0)
                    view_indices[view] = sample_df.index.tolist()
                    all_pred_files.extend(pred_files_for_view)
            
            # Process ensemble data
            if all_pred_files:
                input_dfs_list, keypoint_names = format_data(
                    input_source=all_pred_files,
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
                    result_df = results_dfs[view]
                    base_files = os.listdir(os.path.join(seed_dirs[0], inference_dir))
                    csv_files = [f for f in base_files if view in f]
                    
                    if csv_files:
                        base_name = csv_files[0]
                        preds_file = os.path.join(output_dir, base_name)
                        result_df.to_csv(preds_file)
                        print(f"Saved ensemble {mode} predictions for {view} view to {preds_file}")
        else:
            # Process each view separately
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
        quantile_keep_pca= quantile_keep_pca, #quantile_keep_pca
        camera_names = views,
        s_frames = [(None,None)], # Keemin wil fix 
        avg_mode = avg_mode,
        var_mode = var_mode,
        inflate_vars = True,
        inflate_vars_kwargs = inflate_vars_kwargs,
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


