import os
import numpy as np
import pandas as pd
import traceback

from typing import List, Tuple, Dict, Optional, Union, Any, Callable
from omegaconf import DictConfig
from pathlib import Path

from lightning_pose.utils import io as io_utils
from lightning_pose.utils.scripts import (
    compute_metrics,
    compute_metrics_single,
)

from lp3d_analysis.io import find_sequence_dir


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

def concatenate_csv_files(files: List[str], is_marker_file: bool = True) -> pd.DataFrame:
    """Concatenate multiple CSV files into a single DataFrame with proper indexing.
    
    Args:
        files: List of CSV file paths to concatenate
        is_marker_file: If True, reads as marker file with multi-level headers.
                        If False, reads as bbox file with single-level header.
        
    Returns:
        Concatenated DataFrame
    """
    concat_df = pd.DataFrame()
    for file in files:
        # Read differently based on file type
        if is_marker_file:
            df = pd.read_csv(file, header=[0, 1, 2], index_col=0)
            df = io_utils.fix_empty_first_row(df)
        else:
            df = pd.read_csv(file, index_col=0)
        
        # Ensure unique indices
        if not concat_df.empty:
            next_index = concat_df.index[-1] + 1
            df.index = df.index + next_index
        
        concat_df = pd.concat([concat_df, df]) if not concat_df.empty else df
    
    return concat_df   
    
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
            # snippet_length = 51  # Typical snippet length
            snippet_length = concat_df.shape[0] // len(csv_files)
            print(f"Snippet length is : {snippet_length}")
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



# -----------------------------------------------------------------------------
# Functions for dealing with crop and Zoom datasets 
# -----------------------------------------------------------------------------

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

    # Inside the loop for columns
    for col in csv_data.columns:
        if col[-1] in ("x", "y"):
            vals = csv_data[col].copy()  # Create a copy to avoid chained operations
            coord_type = col[-1]  # 'x' or 'y'
            
            if mode == "subtract":
                # First apply the subtraction
                vals = vals - bbox_data[coord_type]
                # Then apply scaling if dimensions are provided
                if coord_type == "x" and img_width:
                    vals = (vals / bbox_data["w"]) * img_width
                elif coord_type == "y" and img_height:
                    vals = (vals / bbox_data["h"]) * img_height
            else:  # mode == "add"
                # First apply scaling if dimensions are provided
                if coord_type == "x" and img_width:
                    vals = (vals / img_width) * bbox_data["w"]
                elif coord_type == "y" and img_height:
                    vals = (vals / img_height) * bbox_data["h"]
                # Then add the offset
                vals = vals + bbox_data[coord_type]
            
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


