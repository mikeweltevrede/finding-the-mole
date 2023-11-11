from types import ModuleType

import polars as pl


class DataExtractor:
    """Data Extractor to read data using an input IO library."""

    def __init__(self, io_library: ModuleType = None) -> None:
        """Initialize DataExtractor.

        The `io_library` argument can be set to determine in which way to load data. Load data by calling
        `self.io.<desired_load_function>`.

        Args:
            io_library: Library with which to load data. If not set (default), use `polars.io`.
        """
        self.io = io_library or pl.io
