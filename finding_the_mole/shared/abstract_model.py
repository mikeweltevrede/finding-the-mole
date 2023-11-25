import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import numpy as np


class Model(ABC):
    def __init__(self: Self) -> None:
        """Abstract model."""
        self._fitted = False

    @abstractmethod
    def fit(self, **kwargs) -> Self:
        """Method to fit model on data. Should set `self._fitted` to `True` after being run.

        Args:
            **kwargs: Keyword arguments can be implemented by concrete methods, e.g. depending on whether it is a
                supervised or unsupervised model it could need a target value or not.

        Returns:
            Fitted model.
        """

    @abstractmethod
    def predict(self: Self, **kwargs) -> np.ndarray[float]:
        """Method to predict on data with the model. Should check that `self._fitted` is `True` before being run.

        Args:
            **kwargs: Keyword arguments can be implemented by concrete methods, e.g. depending on whether it is a
                supervised or unsupervised model it could need specific arguments.

        Returns:
            Predictions on the data.
        """

    def to_pickle(self: Self, path: Path | os.PathLike | str) -> None:
        """Store model in a pickle file.

        Saves the object into a pickle file at the given path.

        Args:
            path: Path to store the model pickle file at.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with Path.open(path, "wb") as stream:
            pickle.dump(obj=self, file=stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, path: Path | os.PathLike | str) -> Self:
        """Load model from Pickle file.

        Loads the object from a Pickle file at the given path.

        Args:
            path: Path to load the model Pickle file from.

        Raises:
            TypeError: When the loaded object is not a Model.

        Returns:
            The Model object loaded from the Pickle file.
        """
        with Path.open(path, "rb") as stream:
            model = pickle.load(file=stream)

        if not isinstance(model, cls):
            raise TypeError("Loaded object is not a Model")
        return model
