from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Optional

import polars as pl
from dataclass_wizard import YAMLWizard


class DataPreprocessor:
    MAPPER = {"M": 1, "O": 0, "P": 0, "X": -math.inf}
    COL_INFERENCE_EPISODE = "InferenceEpisode"

    @dataclass
    class Context(YAMLWizard):
        """Context dataclass for data preprocessing"""

        index_col: str
        inference_episode: int | str  # TODO: File issue on GitHub that it cannot handle Literal
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

    def _get_max_episode(self, data: pl.DataFrame, prefix_task_cols: str = "Task") -> int:
        """Get maximum episode in the data.

        The maximum episode is determined by considering the task columns, i.e. those prefixed with `prefix_task_cols`.
        By looking at the largest number that comes after `prefix_task_cols`, we can determine how many tasks are in the
        data. Then, looking at the context value `tasks_per_episode`, we determine how many episodes were played. If the
        `tasks_per_episode` does not divide the largest task number, an exception is raised as this is a data quality
        issue.

        Args:
            data: Data to determine the maximum episode for.
            prefix_task_cols: Prefix for the task columns. Combined with an integer to determine the column names.
                Defaults to "Task".

        Returns:
            Maximum episode number in the data.

        Raises:
            RuntimeError: Raised if the context value `tasks_per_episode` does not divide the largest task number.
        """
        task_nums = [int(col.replace(prefix_task_cols, "")) for col in data.columns if prefix_task_cols in col]
        max_episode = (max_task_int := max(task_nums)) / self.context.tasks_per_episode

        if int(max_episode) != max_episode:
            raise RuntimeError(f"{max_task_int=} is not divisible by {self.context.tasks_per_episode=}")
        return int(max_episode)

    @classmethod
    def put_cols_at_start(cls, data: pl.DataFrame, starting_cols: list[str]) -> pl.DataFrame:
        """Put certain columns at the start of the DataFrame.

        Put `starting_cols` in the provided order at the start of the DataFrame. Other columns are kept in the same
        order.

        Args:
            data: Data to reorder.
            starting_cols: Columns (ordered) to put at the start of the DataFrame.

        Returns:
            DataFrame with provided columns at the start.
        """
        return data.select(*starting_cols, pl.exclude(*starting_cols))

    def get_inference_episode(self, data: Optional[pl.DataFrame] = None, prefix_task_cols: str = "Task") -> int:
        """Get episode to do inference on.

        Determine the episode to do inference on as the context value `inference_episode` is that is not `"latest"` and,
        otherwise, get the episode from `self._get_max_episode()`.

        Args:
            data: Data to determine the inference episode for.
            prefix_task_cols: Prefix for the task columns. Combined with an integer to determine the column names.
                Defaults to "Task".

        Returns:
            Episode number used for inference.
        """
        if data is None and self.context.inference_episode == "latest":
            raise ValueError("Argument data can only be None if self.context.inference_episode is not 'latest'")

        return (
            self.context.inference_episode
            if self.context.inference_episode != "latest"
            else self._get_max_episode(data=data, prefix_task_cols=prefix_task_cols)
        )

    def add_inference_episode_column(self, data: pl.DataFrame, prefix_task_cols: str = "Task") -> pl.DataFrame:
        """Add inference episode column.

        Add inference episode column to the data, determined from `self.get_inference_episode()` with name from the
        class attributes.

        Args:
            data: Data to add the inference episode column to.
            prefix_task_cols: Prefix for the task columns. Combined with an integer to determine the column names.
                Defaults to "Task".

        Returns:
            Data with the inference episode column added.
        """
        inference_episode = self.get_inference_episode(data=data, prefix_task_cols=prefix_task_cols)
        return data.with_columns(pl.lit(inference_episode).cast(pl.Int64).alias(self.COL_INFERENCE_EPISODE))

    def limit_data_to_set(self, data: pl.DataFrame, prefix_task_cols: str = "Task") -> pl.DataFrame:
        """Filters columns in `data` to only keep the task columns to infer for, keeping `self.context.index_col`.

        Using `self.context.tasks_per_episode` and `self.context.inference_episode`, we determine which columns to keep. If
        `self.context.inference_episode` is `"latest"`, we don't need to filter anything.

        Args:
            data: Data to filter columns for.
            prefix_task_cols: Prefix for the task columns. Combined with an integer to determine the column names.
                Defaults to "Task".

        Returns:
            Data with only the columns of the inference set, including the index column.
        """
        inference_episode = self.get_inference_episode(data=data, prefix_task_cols=prefix_task_cols)

        if self.context.inference_episode != "latest":
            # For the non-latest episode, we need to only keep the task columns for the desired episode.
            tasks_to_keep = range(1, inference_episode * self.context.tasks_per_episode + 1)
            data = data.select(self.context.index_col, *(f"{prefix_task_cols}{num}" for num in tasks_to_keep))

        return data
