import polars as pl
import pytest
from assertpy import assert_that

from finding_the_mole.shared.data_preprocessing import DataPreprocessor


class TestDataPreprocessor:
    @pytest.fixture()
    def data_preprocessor(self) -> DataPreprocessor:
        return DataPreprocessor()

    @pytest.mark.parametrize("m", ["M", "m"])
    def test_convert_letters_to_ints_converts_m_to_1_with_default_mapper_case_insensitive(
        self, data_preprocessor: DataPreprocessor, m: str
    ):
        data = pl.DataFrame(data=[dict(Task1=m)])
        actual = data_preprocessor.map_values(data=data)

        assert_that(actual[0, "Task1"]).is_equal_to(1)

    @pytest.mark.parametrize("x", ["X", "x"])
    def test_convert_letters_to_ints_converts_x_to_none_with_default_mapper_case_insensitive(
        self, data_preprocessor: DataPreprocessor, x: str
    ):
        data = pl.DataFrame(data=[dict(Task1=x)])
        actual = data_preprocessor.map_values(data=data)

        assert_that(actual[0, "Task1"]).is_equal_to(None)

    @pytest.mark.parametrize("o", ["O", "o"])
    def test_convert_letters_to_ints_converts_o_to_0_with_default_mapper_case_insensitive(
        self, data_preprocessor: DataPreprocessor, o: str
    ):
        data = pl.DataFrame(data=[dict(Task1=o)])
        actual = data_preprocessor.map_values(data=data)

        assert_that(actual[0, "Task1"]).is_equal_to(0)

    @pytest.mark.parametrize("p", ["P", "p"])
    def test_convert_letters_to_ints_converts_p_to_0_with_default_mapper_case_insensitive(
        self, data_preprocessor: DataPreprocessor, p: str
    ):
        data = pl.DataFrame(data=[dict(Task1=p)])
        actual = data_preprocessor.map_values(data=data)

        assert_that(actual[0, "Task1"]).is_equal_to(0)
