import os
import numpy as np
import pandas as pd
import traceback

from typing import List, Tuple, Dict, Optional, Union, Any
from omegaconf import DictConfig
from pathlib import Path

from lightning_pose.utils import io as io_utils
from lightning_pose.utils.scripts import (
    compute_metrics,
    compute_metrics_single,
)

def setup_ensemble_dirs(
    base_dir: str, 
    model_type: str, 
    n_labels: int, 
    seed_range: tuple[int, int],
    mode: str,
    inference_dir: str
) -> Tuple[str, List[str], str]:
    """
    Set up and return directories for ensemble processing.
    
    Args:
        base_dir: Base directory for all models
        model_type: Type of model being used
        n_labels: Number of labels in the model
        seed_range: Range of seeds to use (start, end) inclusive
        mode: Processing mode
        inference_dir: Directory name for inference results
        
    Returns:
        EnsembleDirs object containing all directory information
    """
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
    """
    Create a mapping of original directories and their CSV files.
    
    Args:
        first_seed_dir: Path to the first seed directory
        inference_dir: Directory containing inference results
        views: List of camera view names
        
    Returns:
        Dictionary mapping directory names to lists of CSV files
    """
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

            csv_files = [
                f
                for f in os.listdir(dir_path)
                if f.endswith(".csv") and not f.endswith("_uncropped.csv")
            ]
            print(f"Found {len(csv_files)} CSV files in {dir_name}")
            original_structure[dir_name] = csv_files
        
    return original_structure

def add_variance_columns(column_structure):
    """
    Add variance columns to the column structure.
    
    Args:
        column_structure: Original MultiIndex column structure
        
    Returns:
        Updated MultiIndex with added variance columns
    """
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

def fill_ensemble_results(
    results_df: pd.DataFrame, 
    ensemble_preds: np.ndarray, 
    ensemble_vars: np.ndarray, 
    ensemble_likes: np.ndarray, 
    keypoint_names: List[str], 
    column_structure: pd.MultiIndex
) -> None:
    """
    Fill in ensemble results into the DataFrame.
    
    Args:
        results_df: DataFrame to fill with results
        ensemble_preds: Array of ensemble predictions (x,y coordinates)
        ensemble_vars: Array of ensemble variances
        ensemble_likes: Array of ensemble likelihoods
        keypoint_names: List of keypoint names
        column_structure: Column structure of the DataFrame
    """
    for k, bp in enumerate(keypoint_names):
        # Use scorer from column_structure if not in bp; assume first scorer applies
        scorer = bp.split('/', 1)[0] if '/' in bp else column_structure.levels[0][0]
        bodypart = bp.split('/', 1)[1] if '/' in bp else bp

        # Fill coordinates and likelihood
        results_df.loc[:, (scorer, bodypart, 'x')] = ensemble_preds[:, k, 0]
        results_df.loc[:, (scorer, bodypart, 'y')] = ensemble_preds[:, k, 1]
        results_df.loc[:, (scorer, bodypart, 'likelihood')] = ensemble_likes[:, k]
        results_df.loc[:, (scorer, bodypart, 'x_ens_var')] = ensemble_vars[:, k, 0]
        results_df.loc[:, (scorer, bodypart, 'y_ens_var')] = ensemble_vars[:, k, 1]
    
    
def extract_ood_frame_predictions(
    cfg_lp: DictConfig,
    data_dir: str,
    results_dir: str,
    overwrite: bool,
    video_dir: str,
) -> None:
    """
    Extract out-of-distribution frame predictions from snippets.
    
    Args:
        cfg_lp: Configuration dictionary
        data_dir: Directory containing input data
        results_dir: Directory for output results
        overwrite: Whether to overwrite existing files
        video_dir: Directory name for video files
    """
    new_csv_files = [f for f in os.listdir(data_dir) if f.endswith('_new.csv')]
    print(f"the new csv files are {new_csv_files}")

    for csv_file in new_csv_files:
        # load original csv 
        file_path = os.path.join(data_dir, csv_file)
        print(f"the file path is {file_path}")
        original_df = pd.read_csv(file_path, header=[0,1,2], index_col=0)
        original_df = io_utils.fix_empty_first_row(original_df) 
        original_index = original_df.index  
        

        # Debug the first frame in the index
        print(f"Original index has {len(original_index)} entries")
        if len(original_index) > 0:
            first_frame = original_index[0]
            print(f"First frame in index: {first_frame}")

        # load each of the new csv files and iterate through the index 
        prediction_name = '_'.join(csv_file.split('_')[1:])
        preds_file = os.path.join(results_dir, video_dir , f'predictions_{prediction_name}') 
        #print(f"the preds file is {preds_file}")
        if os.path.exists(preds_file) and not overwrite:
            print(f'Predictions file {preds_file} already exists. Skipping.')
            continue
        
        results_list = []
        #file_path = os.path.join(data_dir, csv_file)
        #df = pd.read_csv(file_path, header=[0,1,2], index_col=0)
        
        # for img_path in df.index:
        for img_path in original_index:
            # process the paths 
            relative_img_path = '/'.join(img_path.split('/')[1:]) # removed 'labeled-data/'
            snippet_path = relative_img_path.replace('png', 'mp4')

            # Load the 51-frame csv file 
            snippet_file = os.path.join(results_dir, video_dir , snippet_path.replace('mp4', 'csv'))
            if os.path.exists(snippet_file):
                snippet_df = pd.read_csv(snippet_file, header=[0,1,2], index_col=0)

                # extract center frame 
                frame_count = snippet_df.shape[0]
                if frame_count % 2 == 0:
                    print(f"Warning: Snippet {snippet_file} has even number of frames: {frame_count}")
                    continue
                idx_frame = int(np.floor(frame_count / 2))
                # assert snippet_df.shape[0] % 2 != 0 # ensure odd number of frames
                # idx_frame = int(np.floor(snippet_df.shape[0] / 2))
                
                # create results with original image path as index 
                # result = snippet_df[snippet_df.index == idx_frame].rename(index={idx_frame: img_path})
                result = snippet_df.iloc[[idx_frame]].rename(index={idx_frame: img_path})
                results_list.append(result)

        # combine all results 
        if results_list:
            results_df = pd.concat(results_list)
            #results_df.sort_index(inplace=True)
            results_df = results_df.reindex(original_index)

            # Add "set" column so this df is interpreted as labeled data predictions
            results_df.loc[:,("set", "", "")] = "train"
            results_df.to_csv(preds_file)
            print(f'Saved predictions to {preds_file}')

        
            try:
                compute_metrics_single(
                    cfg=cfg_lp,
                    labels_file=os.path.join(cfg_lp.data.data_dir, csv_file),
                    preds_file=preds_file,
                    data_module=None
                )
                print(f"Succesfully computed metrics for {preds_file}")
            except Exception as e:
                print(f"Error computing metrics\n{e}")
                print(traceback.format_exc())


def find_sequence_dir(output_dir: str, sequence_name: str, view: str) -> Optional[str]:
    """
    Find the appropriate directory for a sequence.
    
    Args:
        output_dir: Base output directory
        sequence_name: Name of the sequence
        view: Camera view name
        
    Returns:
        Directory name if found, None otherwise
    """
    # Check different possible naming patterns
    possible_patterns = [
        sequence_name,
        f"{sequence_name}_{view}",
        f"{view}_{sequence_name}"
    ]
    
    for pattern in possible_patterns:
        for dir_name in os.listdir(output_dir):
            if pattern in dir_name:
                return dir_name
    
    return None


def process_concatenated_snippets(
    output_dir: str,
    seed_dirs: List[str],
    inference_dir: str,
    img_paths_by_dir: Dict[str, List[str]],
    view: str
) -> List[pd.DataFrame]:
    """
    Process concatenated snippets for a view.
    
    Args:
        output_dir: Directory for output files
        seed_dirs: List of seed directories
        inference_dir: Directory name for inference results
        img_paths_by_dir: Dictionary mapping directories to image paths
        view: Camera view name
        
    Returns:
        List of DataFrames with processed results
    """
    results_list = []
    
    for sequence_name, img_paths in img_paths_by_dir.items():
        # Find the appropriate directory name pattern for this sequence
        sequence_dir = find_sequence_dir(output_dir, sequence_name, view)
        if not sequence_dir:
            continue
        
        concat_file = os.path.join(output_dir, sequence_dir, "concatenated_sequence.csv")
        if not os.path.exists(concat_file):
            continue
            
        try:
            # Load the concatenated CSV with multi-index columns
            concat_df = pd.read_csv(concat_file, header=[0, 1, 2], index_col=0)
            concat_df = io_utils.fix_empty_first_row(concat_df)
            
            # Find first seed directory with this sequence
            first_seed_dir = next(
                (os.path.join(sd, inference_dir, sequence_dir) 
                 for sd in seed_dirs 
                 if os.path.exists(os.path.join(sd, inference_dir, sequence_dir))),
                None
            )
            
            if not first_seed_dir:
                continue
                
            # Get sorted CSV files (original snippets)
            csv_files = sorted(
                [f for f in os.listdir(first_seed_dir) 
                 if f.endswith(".csv") and not f.endswith("_uncropped.csv")],
                key=lambda x: int(os.path.splitext(x)[0].replace("img", ""))
            )
            
            # Map each original image path to its position in the original sequence of files
            img_to_position = {}
            for i, csv_file in enumerate(csv_files):
                base_name = os.path.splitext(csv_file)[0]
                img_path = f"labeled-data/{sequence_name}/{base_name}.png"
                if img_path in img_paths:
                    img_to_position[img_path] = i
        
            # Sort img_paths by their position in the sequence
            img_paths_ordered = sorted(img_paths, key=lambda x: img_to_position.get(x, float('inf')))
            
            # Process each snippet (51 frames each) and extract the center frame
            snippet_length = 51  # Typical snippet length
            for i, img_path in enumerate(img_paths_ordered):
                # Calculate the center frame index for this snippet
                center_idx = i * snippet_length + (snippet_length // 2)
                if center_idx < concat_df.shape[0]:
                    # Extract the center frame and set index
                    center_frame = concat_df.iloc[[center_idx]]
                    center_frame.index = [img_path]
                    results_list.append(center_frame)
        
        except Exception as e:
            print(f"Error processing concatenated sequence {concat_file}: {e}")
            continue
            
    return results_list


def process_individual_snippets(
    output_dir: str,
    original_index: pd.Index
) -> List[pd.DataFrame]:
    """
    Process individual snippets for a view.
    
    Args:
        output_dir: Directory for output files
        original_index: Original image index
        
    Returns:
        List of DataFrames with processed results
    """
    results_list = []
    
    for img_path in original_index:
        try:
            sequence_name = img_path.split('/')[1]  # e.g., "180623_000_bot"
            base_filename = os.path.splitext(os.path.basename(img_path))[0]  # e.g., "img107262"
            
            # Get the sequence CSV path
            snippet_file = os.path.join(output_dir, sequence_name, f"{base_filename}.csv")
            
            if not os.path.exists(snippet_file):
                continue
                
            # Load the CSV with multi-index columns
            snippet_df = pd.read_csv(snippet_file, header=[0, 1, 2], index_col=0)
            snippet_df = io_utils.fix_empty_first_row(snippet_df)
            
            # Ensure odd number of frames
            frame_count = snippet_df.shape[0]
            if frame_count % 2 == 0:
                print(f"Warning: Snippet {snippet_file} has even number of frames: {frame_count}")
                continue
                
            idx_frame = int(np.floor(frame_count / 2))  # Get center frame index
            
            # Extract center frame and set index
            center_frame = snippet_df.iloc[[idx_frame]]  # Keep as DataFrame with one row
            center_frame.index = [img_path]
            results_list.append(center_frame)
                
        except Exception as e:
            print(f"Error processing individual snippet for {img_path}: {e}")
            continue
            
    return results_list


def process_final_predictions(
    view: str,
    output_dir: str,
    seed_dirs: List[str],
    inference_dir: str,
    cfg_lp: DictConfig,
    use_concatenated: bool = False
) -> None:
    """
    Process and save final predictions for a view, computing metrics.
    
    Args:
        view: The camera view name
        output_dir: Directory where outputs will be saved
        seed_dirs: List of seed directories
        inference_dir: Directory name containing inference results
        cfg_lp: Configuration dictionary
        use_concatenated: If True, process concatenated snippets; if False, process individual snippets
    """
    # Load original CSV file to get frame paths
    orig_pred_file = os.path.join(seed_dirs[0], inference_dir, f'predictions_{view}_new.csv')
    try:
        original_df = pd.read_csv(orig_pred_file, header=[0, 1, 2], index_col=0)
        original_df = io_utils.fix_empty_first_row(original_df)
        original_index = original_df.index
    except Exception as e:
        print(f"Error loading original prediction file {orig_pred_file}: {e}")
        return
    
    results_list = []
    
    if use_concatenated:
        # Group image paths by their directories for concatenated processing
        img_paths_by_dir = {}
        for img_path in original_index:
            # Get sequence name (directory)
            sequence_name = img_path.split('/')[1]  # e.g., "180623_000_bot"
            if sequence_name not in img_paths_by_dir:
                img_paths_by_dir[sequence_name] = []
            img_paths_by_dir[sequence_name].append(img_path)
        
        results_list = process_concatenated_snippets(
            output_dir=output_dir,
            seed_dirs=seed_dirs,
            inference_dir=inference_dir,
            img_paths_by_dir=img_paths_by_dir,
            view=view
        )
    else:
        # Process individual snippets
        results_list = process_individual_snippets(
            output_dir=output_dir,
            original_index=original_index
        )
    
    if results_list:
        try:
            # Combine all results in original order
            results_df = pd.concat(results_list)
            results_df = results_df.reindex(original_index)  # Reindex to match original
            results_df.loc[:,("set", "", "")] = "train"  # Add "set" column for labeled data predictions
            
            # Save predictions
            preds_file = os.path.join(output_dir, f'predictions_{view}_new.csv')
            results_df.to_csv(preds_file)
            print(f"Saved combined predictions to {preds_file}")
            
            # Compute metrics
            cfg_lp_view = cfg_lp.copy()
            cfg_lp_view.data.csv_file = [f'CollectedData_{view}_new.csv']
            cfg_lp_view.data.view_names = [view]
            
            compute_metrics(cfg=cfg_lp_view, preds_file=[preds_file], data_module=None)
            print(f"Successfully computed metrics for {preds_file}")
        except Exception as e:
            print(f"Error in final processing for view {view}: {e}")
            print(traceback.format_exc())

# def process_final_predictions(
#     view: str,
#     output_dir: str,
#     seed_dirs: List[str],
#     inference_dir: str,
#     cfg_lp: DictConfig,
#     use_concatenated: bool = False
# ) -> None:
#     """
#     Process and save final predictions for a view, computing metrics.
    
#     Args:
#         view: The camera view name.
#         output_dir: Directory where outputs will be saved.
#         seed_dirs: List of seed directories.
#         inference_dir: Directory name containing inference results.
#         cfg_lp: Configuration dictionary.
#         use_concatenated: If True, process concatenated snippets; if False, process individual snippets.
#     """
#     # Load original CSV file to get frame paths
#     orig_pred_file = os.path.join(seed_dirs[0], inference_dir, f'predictions_{view}_new.csv')
#     original_df = pd.read_csv(orig_pred_file, header=[0, 1, 2], index_col=0)
#     original_df = io_utils.fix_empty_first_row(original_df)
#     original_index = original_df.index
    
#     results_list = []
    
#     if use_concatenated:
#         # Group image paths by their directories for concatenated processing
#         image_paths_by_dir = {}
#         for img_path in original_index:
#             # Get sequence name (directory)
#             sequence_name = img_path.split('/')[1]  # e.g., "180623_000_bot"
#             if sequence_name not in image_paths_by_dir:
#                 image_paths_by_dir[sequence_name] = []
#             image_paths_by_dir[sequence_name].append(img_path)
        
#         # Process each directory's frames together
#         for sequence_name, img_paths in image_paths_by_dir.items():
#             # Find the appropriate directory name pattern for this sequence
#             possible_dirs = [
#                 sequence_name,
#                 f"{sequence_name}_{view}",
#                 view + "_" + sequence_name
#             ]
            
#             sequence_dir = next(
#                 (d for pattern in possible_dirs 
#                 for d in os.listdir(output_dir) if pattern in d),
#                 None
#             )
#             if not sequence_dir:
#                 continue
            
#             concat_file = os.path.join(output_dir, sequence_dir, "concatenated_sequence.csv")
            
#             if os.path.exists(concat_file):
#                 try:
#                     # Load the concatenated CSV with multi-index columns
#                     concat_df = pd.read_csv(concat_file, header=[0, 1, 2], index_col=0)
#                     concat_df = io_utils.fix_empty_first_row(concat_df)
                    
#                     # Find first seed directory with this sequence
#                     first_seed_dir = next((os.path.join(sd, inference_dir, sequence_dir) 
#                                          for sd in seed_dirs 
#                                          if os.path.exists(os.path.join(sd, inference_dir, sequence_dir))), None)
                    
#                     if not first_seed_dir:
#                         continue
#                     # Get sorted CSV files (original snippets)
#                     csv_files = sorted(
#                         [f for f in os.listdir(first_seed_dir) 
#                          if f.endswith(".csv") and not f.endswith("_uncropped.csv")],
#                         key=lambda x: int(os.path.splitext(x)[0].replace("img", ""))
#                     ) #  I am not sure that the sort thing makes sense here - check it 
                    
#                     # Map each original image path to its position in the original sequence of files
#                     img_to_position = {}
#                     for i, csv_file in enumerate(csv_files):
#                         base_name = os.path.splitext(csv_file)[0]
#                         img_path = f"labeled-data/{sequence_name}/{base_name}.png"
#                         if img_path in img_paths:
#                             img_to_position[img_path] = i
                
#                     # Sort img_paths by their position in the sequence
#                     img_paths_ordered = sorted(img_paths, key=lambda x: img_to_position.get(x, float('inf')))
                    
#                     # Process each snippet (51 frames each) and extract the center frame
#                     snippet_length = 51  # Typical snippet length
#                     for i, img_path in enumerate(img_paths_ordered):
#                         # Calculate the center frame index for this snippet
#                         center_idx = i * snippet_length + (snippet_length // 2)
#                         if center_idx < concat_df.shape[0]:
#                             # Extract the center frame and set index
#                             center_frame = concat_df.iloc[[center_idx]]
#                             center_frame.index = [img_path]
#                             results_list.append(center_frame)
                
#                 except Exception as e:
#                     pass
#     else:
#         # Process individual snippets (original method) - each snippets is processed independently and no concat
#         for img_path in original_index:
#             sequence_name = img_path.split('/')[1]  # e.g., "180623_000_bot"
#             base_filename = os.path.splitext(os.path.basename(img_path))[0]  # e.g., "img107262"
#             snippet_file = os.path.join(output_dir, sequence_name, f"{base_filename}.csv") # Get the sequence CSV path
            
#             if os.path.exists(snippet_file):
#                 try:
#                     # Load the CSV with multi-index columns
#                     snippet_df = pd.read_csv(snippet_file, header=[0, 1, 2], index_col=0)
#                     snippet_df = io_utils.fix_empty_first_row(snippet_df)
                    
#                     # Ensure odd number of frames
#                     assert snippet_df.shape[0] & 1 == 1, f"Expected odd number of frames, got {snippet_df.shape[0]}"
#                     idx_frame = int(np.floor(snippet_df.shape[0] / 2)) # Get center frame index
                    
#                     # Extract center frame and set index
#                     center_frame = snippet_df.iloc[[idx_frame]]  # Keep as DataFrame with one row
#                     center_frame.index = [img_path]
#                     results_list.append(center_frame)
                    
#                 except Exception as e:
#                     pass
    
#     if results_list:
#         # Combine all results in original order
#         results_df = pd.concat(results_list)
#         results_df = results_df.reindex(original_index) # Reindex to match original
#         results_df.loc[:,("set", "", "")] = "train" # Add "set" column for labeled data predictions
        
#         # Save predictions
#         preds_file = os.path.join(output_dir, f'predictions_{view}_new.csv')
#         results_df.to_csv(preds_file)
        
#         # Compute metrics
#         cfg_lp_view = cfg_lp.copy()
#         cfg_lp_view.data.csv_file = [f'CollectedData_{view}_new.csv']
#         cfg_lp_view.data.view_names = [view]
        
#         try:
#             compute_metrics(cfg=cfg_lp_view, preds_file=[preds_file], data_module=None)
#         except Exception as e:
#             pass