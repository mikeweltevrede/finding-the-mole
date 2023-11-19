import numpy as np
import polars as pl

from finding_the_mole.shared.abstract_entrypoint_inference import AbstractInferenceJob
from finding_the_mole.shared.abstract_model import Model


class InferenceJob(AbstractInferenceJob):
    def launch(self) -> None:
        """Main method of the InferenceJob. Orchestrates all other logic."""
        self.data_extraction()
        self.data_preprocessing()
        self.model_loading()
        self.model_inference()

    def data_extraction(self, **kwargs) -> pl.DataFrame:
        """Data extraction orchestration method of the InferenceJob."""

    def data_preprocessing(self, **kwargs) -> pl.DataFrame:
        """Data preprocessing orchestration method of the InferenceJob."""

    def model_loading(self, **kwargs) -> Model:
        """Model loading orchestration method of the InferenceJob."""

    def model_inference(self, **kwargs) -> np.ndarray[float]:
        """Model inference orchestration method of the InferenceJob."""
