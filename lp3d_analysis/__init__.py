from importlib.metadata import PackageNotFoundError, version

from lp3d_analysis.dataset_info import dataset_info
from lp3d_analysis.distillation import (
    DistillationConfig,
    DistillationPipeline,
    run_distillation_pipeline
)


try:
    __version__ = version("lp3d-analysis")
except PackageNotFoundError:
    # package is not installed
    pass
