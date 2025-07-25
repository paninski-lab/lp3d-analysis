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
        if model.config.is_multi_view():
            # For multiview case, we need to find video files for each view
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
                
                # Group video files by session (same base name but different view suffixes)
                # and find files for each view
                view_names = model.config.cfg.data.view_names
                session_groups = {}
                
                for video_file in video_files:
                    # Extract base session name by removing view suffixes
                    base_name = video_file
                    matched_view = None
                    
                    # Find which view this video file belongs to using regex-like matching
                    # Use the same pattern as collect_video_files_by_view
                    for view_name in view_names:
                        # Check if view name is in the filename with proper delimiters
                        # This matches the logic in collect_video_files_by_view
                        if re.search(rf"(?<!0-9a-zA-Z){re.escape(view_name)}(?![0-9a-zA-Z])", Path(video_file).name):
                            matched_view = view_name
                            # Remove the view name from the base name
                            base_name = re.sub(rf"(?<!0-9a-zA-Z){re.escape(view_name)}(?![0-9a-zA-Z])", "", Path(video_file).name)
                            break
                    
                    if matched_view:
                        if base_name not in session_groups:
                            session_groups[base_name] = {}
                        
                        if matched_view not in session_groups[base_name]:
                            session_groups[base_name][matched_view] = video_file
                        else:
                            print(f"Warning: Multiple files found for view {matched_view} in session {base_name}")
                
                # Process each session that has all required views
                for session_name, view_files in session_groups.items():
                    if len(view_files) == len(view_names):
                        # All views are present for this session
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
                        # Use the first video file to determine the output directory structure
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
                            model.predict_on_video_file_multiview(
                                video_file_per_view=video_files_per_view,
                                output_dir=output_dir,
                                compute_metrics=False,
                            )
                            print(f"Successfully completed inference for session {session_name}")
                        except Exception as e:
                            print(f"Error during inference for session {session_name}: {str(e)}")
                            continue
                    else:
                        print(f"Warning: Session {session_name} missing some views. Found: {list(view_files.keys())}, Required: {view_names}")

        else: # for the single view case 
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
    else:
        print(
            'inference_dirs None, not running inference. To run inference on videos in a given '
            'directory add it to cfg_pipeline.train_networks.inference_dirs'
        )
    return
        