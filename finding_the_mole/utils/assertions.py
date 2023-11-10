def assert_dict_values_have_same_length(data: dict) -> None:
    """Assert that the values in dictionary ``data`` have the same length.

    Args:
        data: Dictionary to check.
    """
    lengths = [len(value) for value in data.values()]
    assert len(unique_lengths := set(lengths)) == 1, f"Mismatch of dictionary value lengths: {unique_lengths}"
