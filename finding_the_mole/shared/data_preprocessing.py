from typing import Any, Hashable

import polars as pl


class DataPreprocessor:
    MAPPER = {"M": 1, "O": 0, "P": 0}

    @classmethod
    def extend_mapper_keys_with_upper_and_lowercase(cls, mapper: dict[str, Any]) -> dict[str, Any]:
        """Extend keys of `mapper` with the upper and lowercase variants.

        Args:
            mapper: Dictionary to extend the keys for.

        Returns:
            Mapper dictionary with the same keys, extended with the lower and uppercase variants of those keys.
        """
        return {
            **mapper,
            **{key.lower(): value for key, value in mapper.items()},
            **{key.upper(): value for key, value in mapper.items()},
        }

    def map_values(self, data: pl.DataFrame, mapper: dict[Hashable, Any] = None) -> pl.DataFrame:
        """Maps values in DataFrame to others according to mapper.

        Converts all cells in `data` according to `mapper`, where all cells equal to the key get the corresponding
        value.

        Args:
            data: Data to convert the values for.
            mapper: Mapper with key equal to what should be replaced and value what the new value is. If not provided,
                the class attribute `MAPPER` is used.

        Returns:
            Same DataFrame with converted values.
        """
        mapper = mapper or DataPreprocessor.MAPPER
        mapper = self.extend_mapper_keys_with_upper_and_lowercase(mapper=mapper)
        return data.select(pl.all().map_dict(mapper))
