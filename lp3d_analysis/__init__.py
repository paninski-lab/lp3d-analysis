from importlib.metadata import PackageNotFoundError, version

from lp3d_analysis.dataset_info import dataset_info


try:
    __version__ = version("lp3d-analysis")
except PackageNotFoundError:
    # package is not installed
    pass
