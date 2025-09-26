"""Functions for training."""
import contextlib
import gc
import re
from pathlib import Path

import torch
import os
from typing import Optional
from lightning_pose.train import train
from lightning_pose.api.model import Model
from omegaconf import DictConfig, OmegaConf

from lp3d_analysis.utils import generate_cropped_csv_file, _get_bbox_path_fn
# from lp3d_analysis.post_process import _get_bbox_path_fn

# TODO: Replace with contextlib.chdir in python 3.11.
@contextlib.contextmanager
def chdir(dir):
    pwd = os.getcwd()
    os.chdir(dir)
    try:
        yield
    finally:
        os.chdir(pwd)

def train_and_infer(
    cfg_lp: DictConfig,
    model_dir: str,
    inference_dirs: Optional[list] = None,
    overwrite: bool = False,
) -> None:

    # Check if model has already been trained
    try:
        model = Model.from_dir(model_dir)
    except OSError:
        model = None

    if model is None or overwrite:
        print('Training model:')
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        with chdir(model_dir):
            model = train(cfg=cfg_lp)
        gc.collect()
        torch.cuda.empty_cache()

    else:
        print('Model already exists, skipping training (overwrite not true).')

    assert model is not None

    if inference_dirs is not None:
        # Determine if this is a multiview model based on model type, not just data structure
        model_type = model.config.cfg.model.model_type
        is_multiview_model = model_type in ['heatmap_multiview', 'heatmap_multiview_transformer']
        
        if is_multiview_model: # Use multiview prediction for multiview models
            
            # For multiview case, we need to find video files for each view
            for video_dir in inference_dirs:
                video_path = os.path.join(cfg_lp.data.data_dir, video_dir)
                video_files = [f for f in os.listdir(video_path) if f.endswith(".mp4")]
                view_names = model.config.cfg.data.view_names
                session_groups = {}

                if not video_files:
                    # Case 1: videos-for-each-labeled-frame structure
                    # Look for subdirectories that match session patterns
                    sub_video_dirs = [
                        sub_dir for sub_dir in os.listdir(video_path)
                        if os.path.isdir(os.path.join(video_path, sub_dir))
                    ]
                    
                    print(f"Found {len(sub_video_dirs)} subdirectories in {video_dir}: {sub_video_dirs[:5]}...")  # Show first 5
                    
                    # Group subdirectories by session (base name without view suffix)
                    for sub_video_dir in sub_video_dirs:
                        sub_path = os.path.join(video_path, sub_video_dir)
                        sub_files = [f for f in os.listdir(sub_path) if f.endswith(".mp4")]
                        
                        if sub_files:  # Only process if there are mp4 files in the subdirectory
                            # Extract base session name by removing view suffixes from directory name
                            base_name = sub_video_dir
                            matched_view = None
                            
                            # Find which view this subdirectory belongs to
                            for view_name in view_names:
                                if re.search(rf"(?<![0-9a-zA-Z]){re.escape(view_name)}(?![0-9a-zA-Z])", sub_video_dir):
                                    matched_view = view_name
                                    # Remove the view name from the base name
                                    base_name = re.sub(rf"(?<![0-9a-zA-Z]){re.escape(view_name)}(?![0-9a-zA-Z])", "", sub_video_dir)
                                    # Clean up extra separators
                                    base_name = re.sub(r'[_-]+', '_', base_name).strip('_-')
                                    break
                            
                            if matched_view:
                                print(f"Matched subdirectory '{sub_video_dir}' to view '{matched_view}', base_name: '{base_name}'")
                                if base_name not in session_groups:
                                    session_groups[base_name] = {}
                                
                                # Store all video files in this view's subdirectory
                                if matched_view not in session_groups[base_name]:
                                    session_groups[base_name][matched_view] = []
                                
                                for sub_file in sub_files:
                                    session_groups[base_name][matched_view].append(os.path.join(sub_video_dir, sub_file))
                            else:
                                print(f"No view match found for subdirectory '{sub_video_dir}' (available views: {view_names})")
                
                else:
                    # Case 2: videos_debug structure (original logic)
                    # Group video files by session (same base name but different view suffixes)
                    for video_file in video_files:
                        # Extract base session name by removing view suffixes
                        base_name = video_file
                        matched_view = None
                        
                        # Find which view this video file belongs to using regex-like matching
                        for view_name in view_names:
                            # Check if view name is in the filename with proper delimiters
                            if re.search(rf"(?<![0-9a-zA-Z]){re.escape(view_name)}(?![0-9a-zA-Z])", Path(video_file).name):
                                matched_view = view_name
                                # Remove the view name from the base name
                                base_name = re.sub(rf"(?<![0-9a-zA-Z]){re.escape(view_name)}(?![0-9a-zA-Z])", "", Path(video_file).name)
                                break
                        
                        if matched_view:
                            if base_name not in session_groups:
                                session_groups[base_name] = {}
                            
                            if matched_view not in session_groups[base_name]:
                                session_groups[base_name][matched_view] = video_file
                            else:
                                print(f"Warning: Multiple files found for view {matched_view} in session {base_name}")
                
                # Process each session that has all required views
                print(f"Found {len(session_groups)} session groups: {list(session_groups.keys())}")
                for session_name, view_files in session_groups.items():
                    print(f"Session '{session_name}' has views: {list(view_files.keys())} (required: {view_names})")
                    if len(view_files) == len(view_names):
                        # All views are present for this session
                        
                        # Handle different file structures
                        if isinstance(list(view_files.values())[0], list):
                            # Case 1: videos-for-each-labeled-frame (multiple files per view)
                            # Process each combination of files across views
                            view_file_lists = [view_files[view_name] for view_name in view_names]
                            
                            # For simplicity, let's process all files in the first view and find corresponding files in other views
                            # This assumes all views have the same set of file names
                            first_view_files = view_file_lists[0]
                            
                            for file_idx, first_view_file in enumerate(first_view_files):
                                video_files_per_view = []
                                missing_views = []
                                file_name = Path(first_view_file).name
                                
                                # Find corresponding files in all views
                                for view_idx, view_name in enumerate(view_names):
                                    view_file_list = view_file_lists[view_idx]
                                    # Find file with same name in this view
                                    corresponding_file = None
                                    for candidate_file in view_file_list:
                                        if Path(candidate_file).name == file_name:
                                            corresponding_file = candidate_file
                                            break
                                    
                                    if corresponding_file:
                                        video_files_per_view.append(os.path.join(video_path, corresponding_file))
                                    else:
                                        missing_views.append(view_name)
                                        break
                                
                                if missing_views:
                                    print(f"Warning: Missing views {missing_views} for file {file_name} in session {session_name}")
                                    continue
                                
                                # Check if inference files already exist for all views
                                all_exist = True
                                for view_idx, (view_name, video_file) in enumerate(zip(view_names, video_files_per_view)):
                                    relative_path = os.path.relpath(video_file, video_path)
                                    view_output_dir = model.model_dir / video_dir / Path(relative_path).parent
                                    video_name = Path(video_file).stem
                                    inference_csv_file = view_output_dir / f"{video_name}.csv"
                                    if not inference_csv_file.exists() or overwrite:
                                        all_exist = False
                                        break
                                
                                if all_exist and not overwrite:
                                    print(f"Skipping inference for {file_name} in session {session_name}, inference files already exist.")
                                    continue
                                
                                try:
                                    print(f"Running inference on {file_name} in session {session_name} with {len(video_files_per_view)} views")
                                    print(f"View names order: {view_names}")
                                    print(f"Video files order: {[Path(f).name for f in video_files_per_view]}")
                                    
                                    # Create output directories for each view to match input structure
                                    view_output_dirs = []
                                    for view_idx, (view_name, video_file) in enumerate(zip(view_names, video_files_per_view)):
                                        relative_path = os.path.relpath(video_file, video_path)
                                        view_output_dir = model.model_dir / video_dir / Path(relative_path).parent
                                        view_output_dir.mkdir(parents=True, exist_ok=True)
                                        view_output_dirs.append(view_output_dir)
                                        print(f"View {view_name} output directory: {view_output_dir}")
                                    
                                    # For videos-for-each-labeled-frame structure, we need to create filenames
                                    # that include view names so collect_video_files_by_view can match them
                                    modified_video_files = []
                                    # Create temp directory in the model output directory, not the data directory
                                    temp_output_dir = model.model_dir / video_dir / "temp_multiview_output"
                                    temp_output_dir.mkdir(parents=True, exist_ok=True)
                                    
                                    for view_name, video_file in zip(view_names, video_files_per_view):
                                        # Create a path with view name embedded for matching
                                        video_path_obj = Path(video_file)
                                        # Insert view name before the file extension
                                        modified_name = f"{video_path_obj.stem}_{view_name}{video_path_obj.suffix}"
                                        # Create modified path in the temp output directory, not the data directory
                                        modified_path = temp_output_dir / modified_name
                                        
                                        # Create symbolic link with the modified name
                                        if modified_path.exists():
                                            modified_path.unlink()
                                        
                                        try:
                                            # Try to create symbolic link
                                            modified_path.symlink_to(video_path_obj)
                                        except (OSError, NotImplementedError):
                                            # Fallback: copy file if symlinks not supported
                                            import shutil
                                            shutil.copy2(video_file, modified_path)
                                        
                                        modified_video_files.append(str(modified_path))
                                    
                                    try:
                                        # Run multiview prediction with temporary output directory
                                        result = model.predict_on_video_file_multiview(
                                            video_file_per_view=modified_video_files,
                                            output_dir=temp_output_dir,
                                            compute_metrics=False,
                                        )
                                        
                                        # Move the prediction files to the correct locations
                                        for view_idx, (view_name, original_video_file) in enumerate(zip(view_names, video_files_per_view)):
                                            # Find the generated CSV file in temp directory
                                            modified_video_name = Path(modified_video_files[view_idx]).stem
                                            temp_csv_file = temp_output_dir / f"{modified_video_name}.csv"
                                            
                                            # Determine the correct output location
                                            original_video_name = Path(original_video_file).stem
                                            target_csv_file = view_output_dirs[view_idx] / f"{original_video_name}.csv"
                                            
                                            if temp_csv_file.exists():
                                                # Move the file to the correct location
                                                import shutil
                                                shutil.move(str(temp_csv_file), str(target_csv_file))
                                                print(f"Moved prediction file to: {target_csv_file}")
                                                
                                                bbox_path = _get_bbox_path_fn(Path(target_csv_file), Path(model.model_dir), Path(cfg_lp.data.data_dir))
                                                if bbox_path.exists():
                                                    uncropped_file = target_csv_file.with_stem(target_csv_file.stem + "_uncropped")
                                                    generate_cropped_csv_file(str(target_csv_file), str(bbox_path), str(uncropped_file), 320, 320, "add")
                                                    print(f"Created uncropped file: {uncropped_file}")
                                            else:
                                                print(f"Warning: Expected prediction file not found: {temp_csv_file}")
                                        
                                        print(f"Successfully completed inference for {file_name} in session {session_name}")
                                        
                                    finally:
                                        # Clean up the temporary symbolic links/copies and temp directory
                                        for modified_file in modified_video_files:
                                            try:
                                                Path(modified_file).unlink()
                                            except FileNotFoundError:
                                                pass
                                        
                                        # Clean up temp output directory
                                        try:
                                            import shutil
                                            if temp_output_dir.exists():
                                                shutil.rmtree(temp_output_dir)
                                        except Exception as cleanup_error:
                                            print(f"Warning: Could not clean up temp directory {temp_output_dir}: {cleanup_error}")
                                                
                                except Exception as e:
                                    print(f"Error during inference for {file_name} in session {session_name}: {str(e)}")
                                    import traceback
                                    traceback.print_exc()
                                    continue
                        
                        else:
                            # Case 2: videos_debug (single file per view, original logic)
                            video_files_per_view = []
                            missing_views = []
                            
                            # Ensure video files are in the same order as view_names
                            for view_name in view_names:
                                if view_name in view_files:
                                    video_files_per_view.append(os.path.join(video_path, view_files[view_name]))
                                else:
                                    missing_views.append(view_name)
                            
                            if missing_views:
                                print(f"Warning: Missing views {missing_views} for session {session_name}")
                                continue
                            
                            # All views found, run prediction
                            # Set output_dir to preserve directory structure similar to single view case
                            first_video_file = video_files_per_view[0]
                            relative_path = os.path.relpath(first_video_file, video_path)
                            output_dir = (model.model_dir / video_dir / Path(relative_path)).parent
                            
                            # Check if inference files already exist for all views
                            all_exist = True
                            for video_file in video_files_per_view:
                                video_name = Path(video_file).stem
                                inference_csv_file = output_dir / f"{video_name}.csv"
                                if not inference_csv_file.exists() or overwrite:
                                    all_exist = False
                                    break
                            
                            if all_exist and not overwrite:
                                print(f"Skipping inference for session {session_name}, all inference files already exist.")
                                continue
                            
                            try:
                                print(f"Running inference on session {session_name} with {len(video_files_per_view)} views")
                                print(f"View names order: {view_names}")
                                print(f"Video files order: {[Path(f).name for f in video_files_per_view]}")
                                print(f"Output directory: {output_dir}")
                                
                                # Ensure output directory exists
                                output_dir.mkdir(parents=True, exist_ok=True)
                                
                                model.predict_on_video_file_multiview(
                                    video_file_per_view=video_files_per_view,
                                    output_dir=output_dir,
                                    compute_metrics=False,
                                )
                                print(f"Successfully completed inference for session {session_name}")
                            except Exception as e:
                                print(f"Error during inference for session {session_name}: {str(e)}")
                                import traceback
                                traceback.print_exc()
                                continue
                    else:
                        print(f"Warning: Session {session_name} missing some views. Found: {list(view_files.keys())}, Required: {view_names}")

        else: # for single-view models (even if multiple views exist in data)
            # Run inference on all InD/OOD videos and compute unsupervised metrics
            for video_dir in inference_dirs:
                video_path = os.path.join(cfg_lp.data.data_dir, video_dir)
                video_files = [f for f in os.listdir(video_path) if f.endswith(".mp4")]

                if not video_files:
                    sub_video_dirs = [
                        sub_dir for sub_dir in os.listdir(video_path)
                        if os.path.isdir(os.path.join(video_path, sub_dir))
                    ]
                    for sub_video_dir in sub_video_dirs:
                        sub_path = os.path.join(video_path, sub_video_dir)
                        sub_files = [f for f in os.listdir(sub_path) if f.endswith(".mp4")]
                        video_files += [os.path.join(sub_video_dir, f) for f in sub_files]

                for video_file in video_files:
                    # Set output_dir such that it preserves directory structure of videos dir:
                    # when video_file == bar.mp4 then output_dir == model_dir / video_dir
                    # when video_file == foo/bar.mp4 then output_dir == model_dir / video_dir / foo
                    output_dir = (model.model_dir / video_dir / video_file).parent
                    inference_csv_file = (output_dir / Path(video_file).with_suffix(".csv").name)
                    if inference_csv_file.exists() and not overwrite:
                        print(f"Skipping inference for {video_file}, Inference file already exists.")
                        continue
                    
                    model.predict_on_video_file(
                        video_file=os.path.join(video_path, video_file),
                        output_dir=output_dir,
                        compute_metrics=False,
                    )
                    
                    bbox_path = _get_bbox_path_fn(Path(inference_csv_file), Path(model.model_dir), Path(cfg_lp.data.data_dir))
                    if bbox_path.exists():
                        uncropped_file = inference_csv_file.with_stem(inference_csv_file.stem + "_uncropped")
                        generate_cropped_csv_file(str(inference_csv_file), str(bbox_path), str(uncropped_file), 320, 320, "add")
                        print(f"Created uncropped file: {uncropped_file}")
    else:
        print(
            'inference_dirs None, not running inference. To run inference on videos in a given '
            'directory add it to cfg_pipeline.train_networks.inference_dirs'
        )
    return