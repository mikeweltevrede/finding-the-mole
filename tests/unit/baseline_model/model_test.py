import math

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
