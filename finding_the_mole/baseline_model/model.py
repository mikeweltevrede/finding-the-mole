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

    def fit(self: Self, data: pl.DataFrame) -> Self:
        """Method to fit model on data.

        It will count the number of times that a candidate has been in a role in which I would expect the Mole to be in.

        Args:
            data: Input data to fit the model on. Index represents the candidate name or index and columns are
                indicating which role the candidate was in.

        Returns:
            Fitted model.
        """
        self._fitted = True
        print(data)  # temp to avoid PCH trigger

    def predict(self, data: pl.DataFrame) -> list[float]:
        """Method to predict on data with the model.

        Predicts Mole probability by converting the fitted model count into the probability of that count compared to
        the total count.

        Args:
            data: Data to predict the Mole probability for.

        Returns:
            List of predictions between 0 and 1.
        """
