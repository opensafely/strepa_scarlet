import argparse
import pathlib
import itertools
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
        max_date = measure_table["date"].max()
        filtered = measure_table[measure_table["date"] != max_date]
        filtered.to_csv(measure_file, index=False, header=True)


if __name__ == "__main__":
    main()
