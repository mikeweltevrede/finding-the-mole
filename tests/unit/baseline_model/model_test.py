import math

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
        baseline_model.fit(data=data, index_col="Name")
        assert_that(baseline_model._fitted).is_true()

    def test_fit_computes_count_per_row(self, baseline_model: BaselineModel):
        data = pl.DataFrame(
            data=[
                dict(Name="John", Task1=1, Task2=1, Task3=0),
                dict(Name="Mary", Task1=0, Task2=1, Task3=0),
            ]
        )
        expected = pl.DataFrame(data=[dict(Name="John", count=2), dict(Name="Mary", count=1)])

        fitted_model = baseline_model.fit(data=data, index_col="Name")
        pl_testing.assert_frame_equal(fitted_model.counts, expected)

    def test_fit_can_handle_infinite_values(self, baseline_model: BaselineModel):
        data = pl.DataFrame(
            data=[
                dict(Name="John", Task1=1, Task2=1, Task3=-math.inf),
                dict(Name="Mary", Task1=0, Task2=1, Task3=0),
            ]
        )
        expected = pl.DataFrame(data=[dict(Name="John", count=-math.inf), dict(Name="Mary", count=1)])

        fitted_model = baseline_model.fit(data=data, index_col="Name")
        pl_testing.assert_frame_equal(fitted_model.counts, expected)
