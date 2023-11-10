import csv
from pathlib import Path

from finding_the_mole.utils import assertions


def write_dict_to_csv(data: dict, path: Path) -> None:
    """Write dictionary ``data`` to a CSV file at ``path``.

    Dictionary keys should be the column names. Values will be the respective column values.

    Args:
        data: Dictionary to write to a CSV file.
        path: Path to the file to write to.
    """
    if not path.name.endswith(".csv"):
        raise ValueError(f"{path=} should end in csv but did not")

    assertions.assert_dict_values_have_same_length(data=data)

    with Path.open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(data.keys())
        writer.writerows(zip(*data.values()))
