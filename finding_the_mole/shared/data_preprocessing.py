import math
from typing import Any, Hashable, Iterable

import polars as pl


class DataPreprocessor:
    MAPPER = {"M": 1, "O": 0, "P": 0, "X": -math.inf}

    @classmethod
    def extend_mapper_str_keys_with_upper_and_lowercase(cls, mapper: dict[str, Any]) -> dict[str, Any]:
        """Extend string-type keys of `mapper` with the upper and lowercase variants.

        Args:
            mapper: Dictionary to extend the string-type keys for.

        Returns:
            Mapper dictionary with the same keys, extended with the lower and uppercase variants of those keys.
        """
        return {
            **mapper,
            **{key.lower(): value for key, value in mapper.items()},
            **{key.upper(): value for key, value in mapper.items()},
        }

    @classmethod
    def map_values(
        cls, data: pl.DataFrame, mapper: dict[Hashable, Any] = None, exclude_cols: Iterable[str] = None
    ) -> pl.DataFrame:
        """Maps values in DataFrame to others according to mapper.

        Converts all cells in `data` according to `mapper`, where all cells equal to the key get the corresponding
        value. Columns provided in `exclude_cols` will not be converted.

        Args:
            data: Data to convert the values for.
            mapper: Mapper with key equal to what should be replaced and value what the new value is. If not provided,
                the class attribute `MAPPER` is used.
            exclude_cols: Column name(s) for which the values will not be converted.

        Returns:
            Same DataFrame with converted values.
        """
        mapper = mapper or DataPreprocessor.MAPPER

        if exclude_cols:
            return data.select(*exclude_cols, pl.all().exclude(*exclude_cols).map_dict(mapper))
        return data.select(pl.all().map_dict(mapper))
