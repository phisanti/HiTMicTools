try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("HiTMicTools")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"
