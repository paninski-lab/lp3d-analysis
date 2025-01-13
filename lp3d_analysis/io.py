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



