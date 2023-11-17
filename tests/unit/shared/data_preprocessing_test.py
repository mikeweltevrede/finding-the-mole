import polars as pl
import pytest
from assertpy import assert_that

from finding_the_mole.shared.data_preprocessing import DataPreprocessor


class TestDataPreprocessor:
    @pytest.fixture()
    def data_preprocessor(self) -> DataPreprocessor:
        return DataPreprocessor()

    def test_extend_mapper_str_keys_with_upper_and_lowercase_retains_existing_key(self):
        mapper = dict(SomeKey=1)
        actual = DataPreprocessor.extend_mapper_str_keys_with_upper_and_lowercase(mapper=mapper)
        assert_that(actual).contains_key("SomeKey")

    def test_extend_mapper_str_keys_with_upper_and_lowercase_adds_uppercase_key(self):
        mapper = dict(SomeKey=1)
        actual = DataPreprocessor.extend_mapper_str_keys_with_upper_and_lowercase(mapper=mapper)
        assert_that(actual).contains_key("SOMEKEY")

    def test_extend_mapper_str_keys_with_upper_and_lowercase_adds_lowercase_key(self):
        mapper = dict(SomeKey=1)
        actual = DataPreprocessor.extend_mapper_str_keys_with_upper_and_lowercase(mapper=mapper)
        assert_that(actual).contains_key("somekey")

    def test_map_values_converts_m_to_1_with_default_mapper(self, data_preprocessor: DataPreprocessor):
        data = pl.DataFrame(data=[dict(Task1="M")])
        actual = data_preprocessor.map_values(data=data)

        assert_that(actual[0, "Task1"]).is_equal_to(1)

    def test_map_values_converts_x_to_none_with_default_mapper(self, data_preprocessor: DataPreprocessor):
        data = pl.DataFrame(data=[dict(Task1="X")])
        actual = data_preprocessor.map_values(data=data)

        assert_that(actual[0, "Task1"]).is_equal_to(None)

    def test_map_values_converts_o_to_0_with_default_mapper(self, data_preprocessor: DataPreprocessor):
        data = pl.DataFrame(data=[dict(Task1="O")])
        actual = data_preprocessor.map_values(data=data)

        assert_that(actual[0, "Task1"]).is_equal_to(0)

    def test_convert_letters_to_ints_converts_p_to_0_with_default_mapper(self, data_preprocessor: DataPreprocessor):
        data = pl.DataFrame(data=[dict(Task1="P")])
        actual = data_preprocessor.map_values(data=data)

        assert_that(actual[0, "Task1"]).is_equal_to(0)
