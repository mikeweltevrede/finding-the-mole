import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import yaml
from dataclass_wizard import YAMLWizard

from finding_the_mole.baseline_model.model import BaselineModel
from finding_the_mole.shared.abstract_entrypoint_inference import AbstractInferenceJob
from finding_the_mole.shared.data_extraction import DataExtractor
from finding_the_mole.shared.data_preprocessing import DataPreprocessor


class InferenceJob(AbstractInferenceJob):
    SCORE_COL = "score"

    @dataclass
    class Context(YAMLWizard):
        """Context dataclass for inference job"""

        read_path: str
        write_path: str
        write_output: bool
        data_file_name: str
        index_col: str

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
        data = self.data_extraction()
        data_preprocessed = self.data_preprocessing(data=data)
        model = self.model_training(data=data_preprocessed)
        results = self.model_inference(model=model)

        # Show the results for all participants that are not eliminated, sorted from highest to lowest score
        with pl.Config() as cfg:
            cfg.set_tbl_rows(len(results))  # Make sure to print all rows
            print(results.filter(pl.col("count") != -math.inf).sort(self.SCORE_COL, descending=True))

    def data_extraction(self) -> pl.DataFrame:
        """Data extraction orchestration method of the TrainingJob."""
        data_extractor = DataExtractor()
        return data_extractor.io.read_csv(source=Path(self.context.read_path) / self.context.data_file_name)

    def data_preprocessing(self, data: pl.DataFrame) -> pl.DataFrame:
        """Data preprocessing orchestration method of the TrainingJob."""
        context = DataPreprocessor.Context.from_yaml_file(file=self.config_path, decoder=yaml.safe_load)
        data_preprocessor = DataPreprocessor(context=context)

        mapper = data_preprocessor.extend_mapper_str_keys_with_upper_and_lowercase(mapper=data_preprocessor.MAPPER)
        data_prepped = data_preprocessor.map_values(data=data, mapper=mapper, exclude_cols=[self.context.index_col])
        data_prepped = data_preprocessor.limit_data_to_set(data=data_prepped)
        data_prepped = data_preprocessor.add_inference_episode_column(data=data_prepped)
        return data_preprocessor.put_cols_at_start(
            data=data_prepped, starting_cols=[self.context.index_col, data_preprocessor.COL_INFERENCE_EPISODE]
        )

    def model_training(self, data: pl.DataFrame) -> BaselineModel:
        """Model training orchestration method of the TrainingJob."""
        model = BaselineModel()
        return model.fit(data=data, exclude_cols=[self.context.index_col, DataPreprocessor.COL_INFERENCE_EPISODE])

    def model_inference(self, **kwargs) -> pl.DataFrame:
        """Model inference orchestration method of the InferenceJob."""
        scores = (model := kwargs["model"]).predict()
        results = model.counts.with_columns(pl.Series(name=self.SCORE_COL, values=scores))

        if self.context.write_output:
            results.write_delta(
                target=Path(self.context.write_path) / "Predictions" / model.__class__.__name__,
                mode="overwrite",
                delta_write_options=dict(partition_by=DataPreprocessor.COL_INFERENCE_EPISODE),
            )

        return results


if __name__ == "__main__":
    job = InferenceJob(config_path=str(Path(__file__).parents[1] / "conf" / "inference_config.yaml"))
    job.launch()
