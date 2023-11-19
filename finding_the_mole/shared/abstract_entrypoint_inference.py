from abc import ABC, abstractmethod

import numpy as np
import polars as pl

from finding_the_mole.shared.abstract_model import Model


class AbstractInferenceJob(ABC):
    MSG_METHOD_NOT_IMPLEMENTED = (
        "Method not implemented in AbstractInferenceJob. If you want to use this, "
        "implement this in concrete inference jobs."
    )

    @abstractmethod
    def launch(self, **kwargs) -> None:
        """Main method of the InferenceJob. Orchestrates all other logic."""

    def data_extraction(self, **kwargs) -> pl.DataFrame:
        """Data extraction orchestration method of the InferenceJob."""
        raise NotImplementedError(self.MSG_METHOD_NOT_IMPLEMENTED)

    def data_preprocessing(self, **kwargs) -> pl.DataFrame:
        """Data preprocessing orchestration method of the InferenceJob."""
        raise NotImplementedError(self.MSG_METHOD_NOT_IMPLEMENTED)

    def model_loading(self, **kwargs) -> Model:
        """Model loading orchestration method of the InferenceJob."""
        raise NotImplementedError(self.MSG_METHOD_NOT_IMPLEMENTED)

    def model_inference(self, **kwargs) -> np.ndarray[float]:
        """Model inference orchestration method of the InferenceJob."""
        raise NotImplementedError(self.MSG_METHOD_NOT_IMPLEMENTED)
