import os
import pickle
import numpy as np
import pandas as pd
import re
import yaml

from pathlib import Path
from omegaconf import DictConfig
from typing import List, Tuple, Dict, Optional, Union, Any

from lightning_pose.utils import io as io_utils


def load_cfgs(config_file: str):
    # Load pipeline config file
    with open(config_file, "r") as file:
        cfg_pipe = yaml.safe_load(file)
    cfg_pipe = DictConfig(cfg_pipe)

    # Load lightning pose config file from the path specified in pipeline config
    lightning_pose_config_path = cfg_pipe.get("lightning_pose_config")
    with open(lightning_pose_config_path, "r") as file:
        lightning_pose_cfg = yaml.safe_load(file)

    cfg_lp = DictConfig(lightning_pose_cfg)
    return cfg_pipe, cfg_lp

def load_lightning_pose_cfg(config_path: str):
    """
    Load the lightning pose configuration file.

    Args:
        config_path (str): Path to the lightning pose config file.

    Returns:
        DictConfig: The loaded lightning pose configuration as a DictConfig object.
    """
    with open(config_path, "r") as file:
        lightning_pose_cfg = yaml.safe_load(file)

    return DictConfig(lightning_pose_cfg)


def collect_files_by_token(files: list[Path], tokens: list[str]) -> dict[str, list[Path]]:
    """Given a list of files, collects them by presence of token in their filenames.

    Token must separated by the rest of the filename by some non-alphanumeric delimiter.
    E.g. for token "top", mouse_top_3.mp4 is allowed, but mousetop3.mp4 is not allowed."""
    files_by_token: dict[str, list[Path]] = {}
    for token in tokens:
        # Search all the video_files for a match.
        for file in files:
            if re.search(rf"(?<!0-9a-zA-Z){re.escape(token)}(?![0-9a-zA-Z])", file.stem):
                if token not in files_by_token:
                    files_by_token[token] = []
                files_by_token[token].append(file)
        # After the search if nothing was added to dict, there is a problem.
        if token not in files_by_token:
            raise ValueError(f"File not found for token: {token}")

    return files_by_token

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
    # elif include_posterior_variance:
    #     selected_coords.extend(['x_posterior_var', 'y_posterior_var'])
    # elif include_posterior_variance:
    #     selected_coords.extend(['x_posterior_var', 'y_posterior_var'])
    if include_posterior_variance:
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


def collect_csv_files_by_seed(
    view: str, 
    dir_name: str, 
    seed_dirs: List[str], 
    inference_dir: str
) -> Dict[int, List[str]]:
    """Collect CSV files grouped by seed directory."""
    csv_files_by_seed = {}
    
    for seed_idx, seed_dir in enumerate(seed_dirs):
        seed_sequence_dir = os.path.join(seed_dir, inference_dir, dir_name)
        if not os.path.exists(seed_sequence_dir):
            continue
            
        csv_files = [
            os.path.join(seed_sequence_dir, f) for f in os.listdir(seed_sequence_dir)
            if f.endswith(".csv") and not f.endswith("_uncropped.csv")
        ]
        
        if csv_files:
            csv_files = sorted(
                csv_files,
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace("img", ""))
            )
            csv_files_by_seed[seed_idx] = csv_files
            print(f" the order of the files is {csv_files}")
    
    return csv_files_by_seed

def group_directories_by_sequence(views: List[str], seed_dir: str, inference_dir: str) -> Dict:
    """Group directories by sequence across camera views."""
    video_dir = os.path.join(seed_dir, inference_dir)
    sequences = {}

    for view in views:
        view_dirs = [d for d in os.listdir(video_dir) 
                    if view in d and os.path.isdir(os.path.join(video_dir, d))]
        print(f"View {view} dirs: {view_dirs}")
        
        for dir_name in view_dirs:
            # Extract sequence name by removing view identifier
            parts = dir_name.split('_')
            sequence_key = '_'.join([p for p in parts if p not in views])
            
            if sequence_key not in sequences:
                sequences[sequence_key] = {}
            sequences[sequence_key][view] = dir_name
    
    return sequences