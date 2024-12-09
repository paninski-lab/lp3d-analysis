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
