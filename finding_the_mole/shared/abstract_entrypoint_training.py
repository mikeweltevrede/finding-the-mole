from abc import ABC, abstractmethod

import polars as pl

from finding_the_mole.baseline_model.model import BaselineModel


class AbstractTrainingJob(ABC):
    MSG_METHOD_NOT_IMPLEMENTED = (
        "Method not implemented in AbstractTrainingJob. If you want to use this, "
        "implement this in concrete training jobs."
    )

    @abstractmethod
    def launch(self, **kwargs) -> BaselineModel:
        """Main method of the TrainingJob. Orchestrates all other logic."""

    def data_extraction(self, **kwargs) -> pl.DataFrame:
        """Data extraction orchestration method of the TrainingJob."""
        raise NotImplementedError(self.MSG_METHOD_NOT_IMPLEMENTED)

    def data_preprocessing(self, **kwargs) -> pl.DataFrame:
        """Data preprocessing orchestration method of the TrainingJob."""
        raise NotImplementedError(self.MSG_METHOD_NOT_IMPLEMENTED)

    def model_training(self, **kwargs) -> BaselineModel:
        """Model training orchestration method of the TrainingJob."""
        raise NotImplementedError(self.MSG_METHOD_NOT_IMPLEMENTED)
