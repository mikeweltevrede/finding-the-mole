import polars as pl
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
