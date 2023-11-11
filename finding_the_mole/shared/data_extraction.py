from pathlib import Path

import polars as pl


class DataExtractor:
    """Data Extractor to read data into Polars DataFrames"""

    @classmethod
    def read_csv(cls, path: Path, **kwargs) -> pl.DataFrame:
        """Reads CSV file into Polars DataFrame.

        Args:
            path: Path to the CSV data to load.
            **kwargs: Arbitrary keyword arguments to `pl.read_csv`.

        Returns:
            Polars DataFrame with loaded CSV data.
        """
        return pl.read_csv(source=path, **kwargs)
