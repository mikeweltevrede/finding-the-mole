from typing import Self

import polars as pl

from finding_the_mole.shared.abstract_model import Model


class BaselineModel(Model):
    def __init__(self) -> None:
        """Baseline model class.

        Model counts the number of times that a candidate has been in a role in which I would expect the Mole to be in.
        The predicted score will be the probability of that count compared to the total count.
        """
        Model.__init__(self)
        self.counts = None

    def fit(self: Self, data: pl.DataFrame, index_col: str = None) -> Self:
        """Method to fit model on data.

        It will count the number of times that a candidate has been in a role in which I would expect the Mole to be in.

        Args:
            data: Input data to fit the model on. Index represents the candidate name or index and columns are
                indicating which role the candidate was in.
            index_col: Column name which represents an index to compute the count over.

        Returns:
            Fitted model.
        """
        # TODO: Refactor to separate function in data preprocessing
        mapper = {"M": 1, "P": 0, "X": -1e6, "O": 0}
        data = data.select(index_col, pl.all().exclude(index_col).map_dict(mapper))

        counts = data.with_columns(data.select(pl.all().exclude(index_col)).sum(axis=1).alias("count"))
        self.counts = counts.select(index_col, "count")
        self._fitted = True

        return self

    def predict(self, data: pl.DataFrame) -> list[float]:
        """Method to predict on data with the model.

        Predicts Mole probability by converting the fitted model count into the probability of that count compared to
        the total count.

        Args:
            data: Data to predict the Mole probability for.

        Returns:
            List of predictions between 0 and 1.
        """
