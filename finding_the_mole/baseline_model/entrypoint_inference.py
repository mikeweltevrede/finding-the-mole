import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import yaml
from dataclass_wizard import YAMLWizard

from finding_the_mole.baseline_model.model import BaselineModel
from finding_the_mole.shared.abstract_entrypoint_inference import AbstractInferenceJob
from finding_the_mole.shared.abstract_model import Model


class InferenceJob(AbstractInferenceJob):
    SCORE_COL = "score"

    @dataclass
    class Context(YAMLWizard):
        """Context dataclass for inference job"""

        read_path_models: str
        model_pickle_name: str

    def __init__(self, config_path: str) -> None:
        """Entrypoint to the training job.

        Args:
            config_path: Path to the training config. Will be used for this job's context as well as the contexts to all
                orchestrated methods.
        """
        self.config_path = config_path
        self.context = self.Context.from_yaml_file(file=config_path, decoder=yaml.safe_load)

    def launch(self) -> None:
        """Main method of the InferenceJob. Orchestrates all other logic."""
        model = self.model_loading()
        results = self.model_inference(model=model)

        # Show the results for all participants that are not eliminated, sorted from highest to lowest score
        print(results.filter(pl.col("count") != -math.inf).sort(self.SCORE_COL, descending=True))

    def model_loading(self) -> Model:
        """Model loading orchestration method of the InferenceJob."""
        return BaselineModel.from_pickle(
            path=Path(self.context.read_path_models) / "BaselineModel" / self.context.model_pickle_name
        )

    def model_inference(self, **kwargs) -> pl.DataFrame:
        """Model inference orchestration method of the InferenceJob.

        TODO: Store data, split by which episode this is the inference for. Delta file?
        """
        scores = (model := kwargs["model"]).predict()
        return model.counts.with_columns(pl.Series(name=self.SCORE_COL, values=scores))


if __name__ == "__main__":
    job = InferenceJob(config_path=str(Path(__file__).parents[1] / "conf" / "inference_config.yaml"))
    job.launch()
