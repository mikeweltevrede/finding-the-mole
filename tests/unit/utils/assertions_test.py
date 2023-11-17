import pytest

from finding_the_mole.utils.assertions import assert_dict_values_have_same_length


def test_assert_dict_values_have_same_length_passes_on_matching_lengths():
    data = {"col1": list(range(3)), "col2": list(range(3))}
    assert_dict_values_have_same_length(data=data)


def test_assert_dict_values_have_same_length_raises_value_error_on_non_matching_lengths():
    data = {"col1": list(range(3)), "col2": list(range(4))}
    with pytest.raises(ValueError, match="Mismatch of dictionary value lengths"):
        assert_dict_values_have_same_length(data=data)
