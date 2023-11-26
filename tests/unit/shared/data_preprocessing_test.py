import math

import polars as pl
import polars.testing as pl_testing
import pytest
from assertpy import assert_that

from finding_the_mole.shared.data_preprocessing import DataPreprocessor


class TestDataPreprocessor:
    @pytest.fixture()
    def data_preprocessor(self) -> DataPreprocessor:
        context = DataPreprocessor.Context(index_col="Index", inference_episode="latest", tasks_per_episode=3)
        return DataPreprocessor(context=context)

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

    def test_map_values_converts_m_to_1_with_default_mapper(self):
        data = pl.DataFrame(data=[dict(Task1="M")])
        actual = DataPreprocessor.map_values(data=data)

        assert_that(actual[0, "Task1"]).is_equal_to(1)

    def test_map_values_converts_x_to_negative_infinity_with_default_mapper(self):
        data = pl.DataFrame(data=[dict(Task1="X")])
        actual = DataPreprocessor.map_values(data=data)

        assert_that(actual[0, "Task1"]).is_equal_to(-math.inf)

    def test_map_values_converts_o_to_0_with_default_mapper(self):
        data = pl.DataFrame(data=[dict(Task1="O")])
        actual = DataPreprocessor.map_values(data=data)

        assert_that(actual[0, "Task1"]).is_equal_to(0)

    def test_map_values_converts_p_to_0_with_default_mapper(self):
        data = pl.DataFrame(data=[dict(Task1="P")])
        actual = DataPreprocessor.map_values(data=data)

        assert_that(actual[0, "Task1"]).is_equal_to(0)

    @pytest.mark.parametrize("index_value", ["M", "P", "O", "X"])
    def test_map_values_does_not_convert_exclude_cols(self, index_value: str):
        data = pl.DataFrame(data=[dict(Index=index_value, Task1="P")])
        actual = DataPreprocessor.map_values(data=data, exclude_cols=["Index"])

        assert_that(actual[0, "Index"]).is_equal_to(index_value)

    def test_get_max_episode_gets_correct_maximum_for_episode_1(self, data_preprocessor: DataPreprocessor):
        data_preprocessor.context.tasks_per_episode = 3
        data = pl.DataFrame(data=[dict(Index=1, Task1=1, Task2=0, Task3=1)])
        actual = data_preprocessor._get_max_episode(data=data, prefix_task_cols="Task")
        assert_that(actual).is_equal_to(1)

    def test_get_max_episode_gets_correct_maximum_for_episode_2(self, data_preprocessor: DataPreprocessor):
        data_preprocessor.context.tasks_per_episode = 3
        data = pl.DataFrame(data=[dict(Index=1, Task1=1, Task2=0, Task3=1, Task4=1, Task5=1, Task6=1)])
        actual = data_preprocessor._get_max_episode(data=data, prefix_task_cols="Task")
        assert_that(actual).is_equal_to(2)

    def test_get_max_episode_gets_correct_maximum_for_episode_9(self, data_preprocessor: DataPreprocessor):
        data_preprocessor.context.tasks_per_episode = 3
        data = pl.DataFrame(data=[dict(Index=1, Task1=1, Task2=0, Task3=1, Task27=1)])
        actual = data_preprocessor._get_max_episode(data=data, prefix_task_cols="Task")
        assert_that(actual).is_equal_to(9)

    def test_get_max_episode_raises_runtimeerror_when_tasks_per_episode_does_not_divide_largest_task_number(
        self, data_preprocessor: DataPreprocessor
    ):
        data_preprocessor.context.tasks_per_episode = 2
        data = pl.DataFrame(data=[dict(Index=1, Task1=1, Task2=0, Task3=1)])

        with pytest.raises(RuntimeError) as e:
            data_preprocessor._get_max_episode(data=data, prefix_task_cols="Task")
        assert_that(str(e.value)).is_equal_to("max_task_int=3 is not divisible by self.context.tasks_per_episode=2")

    def test_put_cols_at_start_can_put_one_column_at_the_start(self):
        data = pl.DataFrame(data=[dict(Index=1, Task1=1, Task2=0, Task3=1)])
        expected = pl.DataFrame(data=[dict(Task1=1, Index=1, Task2=0, Task3=1)])
        actual = DataPreprocessor.put_cols_at_start(data=data, starting_cols=["Task1"])
        pl_testing.assert_frame_equal(actual, expected)

    def test_put_cols_at_start_can_put_two_columns_at_the_start(self):
        data = pl.DataFrame(data=[dict(Index=1, Task1=1, Task2=0, Task3=1)])
        expected = pl.DataFrame(data=[dict(Task1=1, Task3=1, Index=1, Task2=0)])
        actual = DataPreprocessor.put_cols_at_start(data=data, starting_cols=["Task1", "Task3"])
        pl_testing.assert_frame_equal(actual, expected)

    def test_limit_data_to_set_with_inference_episode_latest_adds_inference_episode_column(
        self, data_preprocessor: DataPreprocessor
    ):
        data_preprocessor.context.inference_episode = "latest"
        df_in = pl.DataFrame(data=[dict(Index="index", Task1=1, Task2=0, Task3=1, Task4=1, Task5=0, Task6=0)])
        actual = data_preprocessor.limit_data_to_set(data=df_in)
        assert_that(actual.columns).contains("InferenceEpisode")

    @pytest.mark.parametrize(("inference_episode", "expected"), [(1, 1), (2, 2), ("latest", 2)])
    def test_limit_data_to_set_adds_inference_episode_column_with_correct_value(
        self, data_preprocessor: DataPreprocessor, inference_episode: int | str, expected: int
    ):
        data_preprocessor.context.inference_episode = inference_episode
        df_in = pl.DataFrame(data=[dict(Index="index", Task1=1, Task2=0, Task3=1, Task4=1, Task5=0, Task6=0)])
        actual = data_preprocessor.limit_data_to_set(data=df_in)
        assert_that(actual[0, "InferenceEpisode"]).is_equal_to(expected)

    def test_limit_data_to_set_with_inference_episode_latest_does_not_filter_original_columns(
        self, data_preprocessor: DataPreprocessor
    ):
        data_preprocessor.context.inference_episode = "latest"
        df_in = pl.DataFrame(data=[dict(Index="index", Task1=1, Task2=0, Task3=1, Task4=1, Task5=0, Task6=0)])
        expected = pl.DataFrame(
            data=[dict(Index="index", InferenceEpisode="6", Task1=1, Task2=0, Task3=1, Task4=1, Task5=0, Task6=0)]
        )
        actual = data_preprocessor.limit_data_to_set(data=df_in)

        pl_testing.assert_frame_equal(actual.select(*df_in.columns), expected.select(*df_in.columns))

    def test_limit_data_to_set_with_inference_episode_to_train_on_1_and_tasks_per_episode_3_keeps_first_3_task_columns(
        self, data_preprocessor: DataPreprocessor
    ):
        data_preprocessor.context.inference_episode = 1
        data_preprocessor.context.tasks_per_episode = 3

        data = pl.DataFrame(data=[dict(Index="index", Task1=1, Task2=0, Task3=1, Task4=1, Task5=0, Task6=0)])
        expected = pl.DataFrame(data=[dict(Index="index", InferenceEpisode=1, Task1=1, Task2=0, Task3=1)])
        actual = data_preprocessor.limit_data_to_set(data=data)

        pl_testing.assert_frame_equal(actual, expected)

    def test_limit_data_to_set_with_inference_episode_to_train_on_1_and_tasks_per_episode_2_keeps_first_2_task_columns(
        self, data_preprocessor: DataPreprocessor
    ):
        data_preprocessor.context.inference_episode = 1
        data_preprocessor.context.tasks_per_episode = 2

        data = pl.DataFrame(data=[dict(Index="index", Task1=1, Task2=0, Task3=1, Task4=1, Task5=0, Task6=0)])
        expected = pl.DataFrame(data=[dict(Index="index", InferenceEpisode=1, Task1=1, Task2=0)])
        actual = data_preprocessor.limit_data_to_set(data=data)

        pl_testing.assert_frame_equal(actual, expected)

    def test_limit_data_to_set_with_inference_episode_to_train_on_2_and_tasks_per_episode_2_keeps_first_4_task_columns(
        self, data_preprocessor: DataPreprocessor
    ):
        data_preprocessor.context.inference_episode = 2
        data_preprocessor.context.tasks_per_episode = 2

        data = pl.DataFrame(data=[dict(Index="index", Task1=1, Task2=0, Task3=1, Task4=1, Task5=0, Task6=0)])
        expected = pl.DataFrame(data=[dict(Index="index", InferenceEpisode=2, Task1=1, Task2=0, Task3=1, Task4=1)])
        actual = data_preprocessor.limit_data_to_set(data=data)

        pl_testing.assert_frame_equal(actual, expected)

    def test_limit_data_to_set_with_inference_episode_to_train_on_2_and_tasks_per_episode_1_keeps_first_2_task_columns(
        self, data_preprocessor: DataPreprocessor
    ):
        data_preprocessor.context.inference_episode = 2
        data_preprocessor.context.tasks_per_episode = 1

        data = pl.DataFrame(data=[dict(Index="index", Task1=1, Task2=0, Task3=1, Task4=1, Task5=0, Task6=0)])
        expected = pl.DataFrame(data=[dict(Index="index", InferenceEpisode=2, Task1=1, Task2=0)])
        actual = data_preprocessor.limit_data_to_set(data=data)

        pl_testing.assert_frame_equal(actual, expected)

    def test_limit_data_to_set_allows_for_custom_prefix_task_cols(self, data_preprocessor: DataPreprocessor):
        df_in = pl.DataFrame(data=[dict(Index="index", T1=1, T2=0, T3=1, T4=1, T5=0, T6=0)])
        expected = pl.DataFrame(data=[dict(Index="index", InferenceEpisode=2, T1=1, T2=0, T3=1, T4=1, T5=0, T6=0)])
        actual = data_preprocessor.limit_data_to_set(data=df_in, prefix_task_cols="T")

        pl_testing.assert_frame_equal(actual, expected)
