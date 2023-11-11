import polars as pl
from assertpy import assert_that

from finding_the_mole.shared.data_extraction import DataExtractor


class TestDataExtractor:
    def test_data_extractor_io_defaults_to_polars_io(self):
        assert_that(DataExtractor().io).is_equal_to(pl.io)

    def test_data_extractor_io_equals_polars_io_when_io_library_is_none(self):
        assert_that(DataExtractor(io_library=None).io).is_equal_to(pl.io)
