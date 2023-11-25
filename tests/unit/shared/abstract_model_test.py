import pickle
from pathlib import Path
from typing import Optional
from unittest import mock

import pytest
from assertpy import assert_that

from finding_the_mole.shared.abstract_model import Model


class TestAbstractModel:
    class DummyModel(Model):
        def fit(self) -> None:
            ...

        def predict(self, data: Optional = None) -> None:
            ...

    @pytest.fixture()
    def model(self) -> Model:
        return self.DummyModel()

    def test_initializes_attribute_fitted_to_false(self, model: Model):
        assert_that(model._fitted).is_false()

    def test_to_pickle_creates_pickle_file(self, model: Model, tmp_path: Path):
        path_pickle = tmp_path / "test_pickle.pkl"
        model.to_pickle(path=path_pickle)
        assert_that(Path.exists(path_pickle)).is_true()

    def test_to_pickle_dumps_self(self, model: Model, tmp_path: Path):
        with mock.patch("pickle.dump") as mocked:
            model.to_pickle(path=tmp_path / "test_pickle.pkl")
            mocked.assert_called_once_with(**dict(obj=model, file=mock.ANY, protocol=mock.ANY))

    def test_to_pickle_stores_using_highest_protocol(self, model: Model, tmp_path: Path):
        with mock.patch("pickle.dump") as mocked:
            model.to_pickle(path=tmp_path / "test_pickle.pkl")
            mocked.assert_called_once_with(**dict(obj=mock.ANY, file=mock.ANY, protocol=pickle.HIGHEST_PROTOCOL))

    def test_from_pickle_reads_model(self, model: Model, tmp_path: Path):
        with Path.open(tmp_file := tmp_path / "model.pkl", "wb") as stream:
            pickle.dump(obj=model, file=stream, protocol=pickle.HIGHEST_PROTOCOL)

        loaded_model = Model.from_pickle(path=tmp_file)
        assert_that(loaded_model).is_instance_of(model.__class__)

    def test_from_pickle_raises_typeerror_if_loaded_pickle_is_not_a_model(self, tmp_path: Path):
        class NotAModel:
            ...

        tmp_file = tmp_path / "tmp_file.txt"
        tmp_file.touch()

        with mock.patch("pickle.load", return_value=NotAModel()):
            with pytest.raises(TypeError) as e:
                Model.from_pickle(path=tmp_file)
            assert_that(str(e.value)).is_equal_to("Loaded object is not a Model")
