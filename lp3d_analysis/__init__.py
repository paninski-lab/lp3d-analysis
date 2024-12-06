from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lp3d-analysis")
except PackageNotFoundError:
    # package is not installed
    pass
