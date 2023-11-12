from typing import Optional

import pytest
from assertpy import assert_that

from finding_the_mole.shared.abstract_model import Model


class TestAbstractModel:
    @pytest.fixture()
    def model(self) -> Model:
        class DummyModel(Model):
            def fit(self) -> None:
                ...

            def predict(self, data: Optional = None) -> None:
                ...

        return DummyModel()

    def test_initializes_attribute_fitted_to_false(self, model: Model):
        assert_that(model._fitted).is_false()
