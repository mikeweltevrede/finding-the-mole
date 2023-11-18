from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Hashable, Iterable

import polars as pl
from dataclass_wizard import YAMLWizard


class DataPreprocessor:
    MAPPER = {"M": 1, "O": 0, "P": 0, "X": -math.inf}

    @dataclass
    class Context(YAMLWizard):
        """Context dataclass for data preprocessing"""

        index_col: str
        num_episodes_to_train_on: int | str  # TODO: File issue on GitHub that it cannot handle Literal
        tasks_per_episode: int = 3

    def __init__(self, context: DataPreprocessor.Context) -> None:
        """Data preprocessing class

        Args:
            context: DataPreprocessor context dataclass with configuration values.
        """
        self.context = context

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

    def limit_data_to_train_set(self, data: pl.DataFrame, prefix_task_cols: str = "Task") -> pl.DataFrame:
        """Filters columns in `data` to only keep the task columns to train on, keeping `self.context.index_col`.

        Using `self.context.tasks_per_episode` and `self.context.num_episodes_to_train_on`, we determine which columns
        to keep. If `self.context.num_episodes_to_train_on` is `"all"`, we don't need to filter anything.

        Args:
            data: Data to filter columns to train set for.
            prefix_task_cols: Prefix for the task columns. Combined with an integer to determine the column names.

        Returns:
            Data with only the columns of the train set, including the index column.
        """
        if self.context.num_episodes_to_train_on == "all":
            return data

        tasks_to_keep = range(1, self.context.num_episodes_to_train_on * self.context.tasks_per_episode + 1)
        return data.select(self.context.index_col, *(f"{prefix_task_cols}{num}" for num in tasks_to_keep))
