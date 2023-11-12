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
        baseline_model.fit(data=pl.DataFrame())

        assert_that(baseline_model._fitted).is_true()

    def test_fit_computes_count_of_m_per_row(self, baseline_model: BaselineModel):
        data = pl.DataFrame(
            data=[
                dict(Name="John", Task1="M", Task2="M", Task3="O"),
                dict(Name="Mary", Task1="P", Task2="M", Task3="O"),
            ]
        )
        expected = pl.DataFrame(data=[dict(Name="John", count=2.0), dict(Name="Mary", count=1.0)])

        fitted_model = baseline_model.fit(data=data, index_col="Name")
        pl_testing.assert_frame_equal(fitted_model.counts, expected)
