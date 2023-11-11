import tempfile
from pathlib import Path

import polars as pl
import pytest
from polars import testing as pl_testing

from finding_the_mole.shared.data_extraction import DataExtractor
from tests.utils import io_utils


class TestDataExtractor:
    @pytest.fixture()
    def data_extractor(self) -> DataExtractor:
        return DataExtractor(io_library=pl.io)

    def test_data_extractor_load_csv_outputs_correct_polars_df(self, data_extractor: DataExtractor):
        data = {"col1": list(range(3)), "col2": list(range(3))}
        expected = pl.DataFrame(data)

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "data.csv"
            io_utils.write_dict_to_csv(data=data, path=data_path)
            actual = data_extractor.io.read_csv(data_path)

        pl_testing.assert_frame_equal(actual, expected)
