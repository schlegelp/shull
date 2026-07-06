from importlib.metadata import version, PackageNotFoundError

from shull import _shull

from .delaunay import *

try:
    __version__ = version("shull")
except PackageNotFoundError:  # not installed, e.g. running from source tree
    __version__ = "0.0.0+unknown"
