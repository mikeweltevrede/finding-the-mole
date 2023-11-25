import math
import pickle
from pathlib import Path
from unittest import mock

import numpy as np
import numpy.testing as np_testing
import polars as pl
import polars.testing as pl_testing
import pytest
from assertpy import assert_that

from finding_the_mole.baseline_model.model import BaselineModel


class TestBaselineModel:
    @pytest.fixture()
    def baseline_model(self) -> BaselineModel:
        return BaselineModel()

    def test_fit_sets_fitted_to_true(self, baseline_model: BaselineModel):
        data = pl.DataFrame(data=[dict(Name="John", Task1=1, Task2=1, Task3=0)])
        baseline_model.fit(data=data, exclude_cols=["Name"])
        assert_that(baseline_model._fitted).is_true()

    def test_fit_computes_count_per_row(self, baseline_model: BaselineModel):
        data = pl.DataFrame(
            data=[
                dict(Name="John", Task1=1, Task2=1, Task3=0),
                dict(Name="Mary", Task1=0, Task2=1, Task3=0),
            ]
        )
        expected = pl.DataFrame(data=[dict(Name="John", count=2), dict(Name="Mary", count=1)])

        fitted_model = baseline_model.fit(data=data, exclude_cols=["Name"])
        pl_testing.assert_frame_equal(fitted_model.counts, expected)

    def test_fit_can_handle_infinite_values(self, baseline_model: BaselineModel):
        data = pl.DataFrame(
            data=[
                dict(Name="John", Task1=1, Task2=1, Task3=-math.inf),
                dict(Name="Mary", Task1=0, Task2=1, Task3=0),
            ]
        )
        expected = pl.DataFrame(data=[dict(Name="John", count=-math.inf), dict(Name="Mary", count=1)])

        fitted_model = baseline_model.fit(data=data, exclude_cols=["Name"])
        pl_testing.assert_frame_equal(fitted_model.counts, expected)

    def test_fit_can_handle_empty_exclude_cols(self, baseline_model: BaselineModel):
        data = pl.DataFrame(
            data=[
                dict(Task1=1, Task2=1, Task3=0),
                dict(NTask1=0, Task2=1, Task3=0),
            ]
        )
        expected = pl.DataFrame(data=[dict(count=2), dict(count=1)])

        fitted_model = baseline_model.fit(data=data, exclude_cols=None)
        pl_testing.assert_frame_equal(fitted_model.counts, expected)

    def test_predict_raises_runtime_error_when_model_not_fitted(self, baseline_model: BaselineModel):
        baseline_model._fitted = False
        with pytest.raises(RuntimeError) as e:
            baseline_model.predict()
        assert_that(str(e.value)).is_equal_to("Model has not been fitted")

    def test_predict_computes_score_proportion(self, baseline_model: BaselineModel):
        expected = np.array([2 / 3, 1 / 3])
        baseline_model.counts = pl.DataFrame(data=[dict(Name="John", count=2), dict(Name="Mary", count=1)])
        baseline_model._fitted = True

        actual = baseline_model.predict()

        np_testing.assert_allclose(actual, expected)

    def test_predict_computes_score_proportion_ignoring_infinite_values(self, baseline_model: BaselineModel):
        expected = np.array([2 / 3, 1 / 3, 0])
        baseline_model.counts = pl.DataFrame(
            data=[dict(Name="John", count=2), dict(Name="Mary", count=1), dict(name="Alex", count=-math.inf)]
        )
        baseline_model._fitted = True

        actual = baseline_model.predict()

        np_testing.assert_allclose(actual, expected)

    def test_to_pickle_creates_pickle_file(self, baseline_model: BaselineModel, tmp_path: Path):
        path_pickle = tmp_path / "test_pickle.pkl"
        baseline_model.to_pickle(path=path_pickle)
        assert_that(Path.exists(path_pickle)).is_true()

    def test_to_pickle_dumps_self(self, baseline_model: BaselineModel, tmp_path: Path):
        with mock.patch("pickle.dump") as mocked:
            baseline_model.to_pickle(path=tmp_path / "test_pickle.pkl")
            mocked.assert_called_once_with(**dict(obj=baseline_model, file=mock.ANY, protocol=mock.ANY))

    def test_to_pickle_stores_using_highest_protocol(self, baseline_model: BaselineModel, tmp_path: Path):
        with mock.patch("pickle.dump") as mocked:
            baseline_model.to_pickle(path=tmp_path / "test_pickle.pkl")
            mocked.assert_called_once_with(**dict(obj=mock.ANY, file=mock.ANY, protocol=pickle.HIGHEST_PROTOCOL))

    def test_from_pickle_reads_model(self, baseline_model: BaselineModel, tmp_path: Path):
        with Path.open(tmp_file := tmp_path / "model.pkl", "wb") as stream:
            pickle.dump(obj=baseline_model, file=stream, protocol=pickle.HIGHEST_PROTOCOL)

        loaded_model: BaselineModel = BaselineModel.from_pickle(path=tmp_file)
        assert_that(loaded_model.counts).is_equal_to(baseline_model.counts)

    def test_from_pickle_raises_typeerror_if_loaded_pickle_is_not_a_model(self, tmp_path: Path):
        class NotAModel:
            ...

        tmp_file = tmp_path / "tmp_file.txt"
        tmp_file.touch()

        with mock.patch("pickle.load", return_value=NotAModel()):
            with pytest.raises(TypeError) as e:
                BaselineModel.from_pickle(path=tmp_file)
            assert_that(str(e.value)).is_equal_to("Loaded object is not a Model")
