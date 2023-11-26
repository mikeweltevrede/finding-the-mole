from typing import Iterable, Self

import numpy as np
import polars as pl

from finding_the_mole.shared.abstract_model import Model


class BaselineModel(Model):
    COUNT_COL = "count"

    def __init__(self) -> None:
        """Baseline model class.

        Model counts the number of times that a candidate has been in a role in which I would expect the Mole to be in.
        The predicted score will be the probability of that count compared to the total count.
        """
        Model.__init__(self)
        self.counts: pl.DataFrame = None

    def fit(self: Self, data: pl.DataFrame, exclude_cols: Iterable[str] = None) -> Self:
        """Method to fit model on data.

        It will count the number of times that a candidate has been in a role in which I would expect the Mole to be.

        Args:
            data: Input data to fit the model on. Index represents the candidate name or index and columns are
                indicating which role the candidate was in.
            exclude_cols: Column name(s) which should not be included in the row count and could be regarded as the
                index/indices.

        Returns:
            Fitted model.
        """
        if exclude_cols:
            counts = data.with_columns(data.select(pl.all().exclude(*exclude_cols)).sum(axis=1).alias(self.COUNT_COL))
            self.counts = counts.select(*exclude_cols, self.COUNT_COL)
        else:
            counts = data.with_columns(data.select(pl.all()).sum(axis=1).alias(self.COUNT_COL))
            self.counts = counts.select(self.COUNT_COL)
        self._fitted = True

        return self

    def predict(self) -> np.ndarray[float]:
        """Method to predict on data with the model.

        Predicts Mole probability by converting the fitted model count into the probability of that count compared to
        the total count. When the count is (negative or positive) infinite, do not take that row into account.

        Returns:
            Numpy array of predictions between 0 and 1.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted")

        data = self.counts.with_columns(
            pl.when(pl.col(self.COUNT_COL).is_infinite()).then(0).otherwise(pl.col(self.COUNT_COL)).keep_name()
        )
        return np.array(data[self.COUNT_COL] / sum(data[self.COUNT_COL]))
