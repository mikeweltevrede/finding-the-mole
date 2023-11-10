def assert_dict_values_have_same_length(data: dict) -> None:
    """Assert that the values in dictionary ``data`` have the same length.

    Args:
        data: Dictionary to check.
    """
    lengths = [len(value) for value in data.values()]
    if len(set(lengths)) != 1:
        raise ValueError(f"Mismatch of dictionary value lengths: {lengths}")
