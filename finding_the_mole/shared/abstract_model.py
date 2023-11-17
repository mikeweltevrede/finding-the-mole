from abc import ABC, abstractmethod
from typing import Optional, Self

import numpy as np
import polars as pl


class Model(ABC):
    def __init__(self: Self) -> None:
        """Abstract model."""
        self._fitted = False

    @abstractmethod
    def fit(self, **kwargs: Optional[pl.DataFrame]) -> Self:
        """Method to fit model on data. Should set `self._fitted` to `True` after being run.

        Args:
            **kwargs: Keyword arguments can be implemented by concrete methods, e.g. depending on whether it is a
                supervised or unsupervised model it could need a target value or not.

        Returns:
            Fitted model.
        """

    @abstractmethod
    def predict(self: Self, *args, **kwargs) -> np.ndarray[float]:
        """Method to predict on data with the model. Should check that `self._fitted` is `True` before being run.

        Returns:
            Predictions on the data.
        """
