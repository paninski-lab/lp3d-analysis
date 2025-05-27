"""Functions for training."""
import contextlib
import gc
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
        