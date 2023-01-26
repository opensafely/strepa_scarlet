import argparse
import pathlib
import itertools
import pandas
import glob
from report_utils import (
    get_measure_tables,
)


def get_path(*args):
    return pathlib.Path(*args).resolve()


def match_paths(pattern):
    return [get_path(x) for x in glob.glob(pattern)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--measures-files",
        required=True,
        action="append",
        type=match_paths,
        help="Glob pattern for matching one or more measures names",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    measures_files = args.measures_files

    all_files = list(itertools.chain(*measures_files))
    for measure_file in all_files:
        measure_table = get_measure_tables(measure_file)
        filtered_min = measure_table[measure_table["date"] >= pandas.to_datetime("2022-11-11")]
        filtered_max = filtered_min[filtered_min["date"] <= pandas.to_datetime("2023-01-13")]
        filtered_max.to_csv(measure_file, index=False, header=True)


if __name__ == "__main__":
    main()
