import re
from pathlib import Path

import yaml
from omegaconf import DictConfig


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
