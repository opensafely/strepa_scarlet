import argparse
import pathlib
from report_utils import (
    autoselect_labels,
    translate_group,
    get_measure_tables,
    subset_table,
)
import pandas


def round_or_skip(x):
    """
    Try to return a value
    If it fails, return the original value
    """
    try:
        return round(float(x))
    except Exception:
        return x


def count_table(measure_table):
    # Translate name column
    repeated = autoselect_labels(measure_table["name"])
    measure_table["name"] = measure_table["name"].apply(
        lambda x: translate_group(x, x, repeated, autolabel=True)
    )
    table = measure_table.pivot(
        index="date", columns="name", values="numerator"
    )

    # Assert that the patient count is the same for all columns
    denominator = measure_table.pivot(
        index="date", columns="name", values="denominator"
    )
    # Ensure that all denominator columns are the same before selecting one
    assert denominator.eq(denominator.iloc[:, 0], axis=0).all(axis=None)
    denominator = denominator.iloc[:, 0]
    denominator.name = "Patient Count"

    # Add the patient count (denominator) as the first column
    table = pandas.concat([denominator, table], axis=1)
    return table


def get_path(*args):
    return pathlib.Path(*args).resolve()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to single joined measures file",
    )
    measures_group = parser.add_mutually_exclusive_group(required=False)
    measures_group.add_argument(
        "--measures-pattern",
        required=False,
        help="Glob pattern for matching one or more measures names",
    )
    measures_group.add_argument(
        "--measures-list",
        required=False,
        action="append",
        help="A list of one or more measure names",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Name for panel plot",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input_file
    measures_pattern = args.measures_pattern
    measures_list = args.measures_list
    output_dir = args.output_dir
    output_name = args.output_name

    measure_table = get_measure_tables(input_file)

    # Parse the names field to determine which subset to use
    subset = subset_table(measure_table, measures_pattern, measures_list)
    table = count_table(subset)
    table.index.name = table.index.name.title()
    table.index = table.index.strftime("%d-%m-%y")
    # NOTE: applymap is by value (so very slow), but we have a mix of strings
    # and numbers (REDACTED) so pandas.to_numeric will not work
    # and the tables are small
    table = table.applymap(round_or_skip)
    table.to_csv(output_dir / output_name)


if __name__ == "__main__":
    main()
