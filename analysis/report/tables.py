import argparse
import pathlib
from report_utils import (
    autoselect_labels,
    translate_group,
    get_measure_tables,
    subset_table,
)


def count_table(measure_table):
    # TODO: TEMP FIX FOR STREP A SORE THROAT
    measure_table["name"] = measure_table["name"].replace(
        {
            "event_strep_a_sore_throat_rate": "event_sore_throat_tonsillitis_rate"
        }
    )
    # Translate name column
    repeated = autoselect_labels(measure_table["name"])
    measure_table["name"] = measure_table["name"].apply(
        lambda x: translate_group(x, x, repeated, autolabel=True)
    )
    table = measure_table.pivot(
        index="date", columns="name", values="numerator"
    )
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
    table.to_csv(output_dir / output_name)


if __name__ == "__main__":
    main()
