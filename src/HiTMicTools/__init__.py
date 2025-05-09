try:
    import pkg_resources

    __version__ = pkg_resources.get_distribution("HiTMicTools").version
except pkg_resources.DistributionNotFound:
    __version__ = "0.0.0+dev"
