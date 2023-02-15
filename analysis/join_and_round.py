import argparse
import pathlib
import re
import glob
import numpy
import pandas
import itertools
from report.report_utils import round_values


MEASURE_FNAME_REGEX = re.compile(r"measure_(?P<id>\S+)\.csv")


def _check_for_practice(table):
    if "practice" in table.category.values:
        raise (
            AssertionError("Practice-level data should not be in final output")
        )


def _reshape_data(measure_table):
    try:
        assert len(measure_table.columns) < 6
    except AssertionError:
        raise (
            AssertionError("This script only supports one group_by category")
        )
    if (
        len(measure_table.date) == 1
        or measure_table.date[0] != measure_table.date[1]
    ):
        # if sequential rows have different dates, then an individual date's
        # data has not been subdivided by category, and we can assume that
        # group_by = "population"
        # This is also true if there is only one time period and one category
        # Therefore, the numerator and denominator will be the first columns
        numerator = measure_table.columns[0]
        denominator = measure_table.columns[1]
        measure_table["category"] = "population"
        measure_table["group"] = "population"
        group_by = None
        measure_table["name"] = measure_table.attrs["id"]

    else:
        denominator = measure_table.columns[-3]
        numerator = measure_table.columns[-4]
        group_by = measure_table.columns[-5]
        measure_table["category"] = group_by
        measure_table["name"] = measure_table.attrs["id"]

    measure_table.rename(
        columns={
            numerator: "numerator",
            denominator: "denominator",
            group_by: "group",
        },
        inplace=True,
    )
    # Assume we only need the numerator and the denominator
    measure_table.drop(columns=["population"], inplace=True, errors="ignore")
    return measure_table


def _ensure_names(df):
    df.group = df.group.fillna("Missing")
    df.category = df.category.fillna("Missing")
    return df


def _join_tables(tables):
    return pandas.concat(tables)


def get_measure_tables(input_files, exclude_files):
    all_files = set(itertools.chain(*input_files))
    all_exclude = set(itertools.chain(*exclude_files))
    all_files = all_files - all_exclude
    for input_file in all_files:
        measure_fname_match = re.match(MEASURE_FNAME_REGEX, input_file.name)
        if measure_fname_match is not None:
            # The `date` column is assigned by the measures framework.
            measure_table = pandas.read_csv(
                input_file, dtype="str", parse_dates=["date"]
            )

            # We can reconstruct the parameters passed to `Measure` without
            # the study definition.
            measure_table.attrs["id"] = measure_fname_match.group("id")
            yield measure_table


def _round_table(measure_table, round_to, redact=False, redaction_threshold=5):
    
    measure_table.numerator = measure_table.numerator.astype(float)
    measure_table.denominator = measure_table.denominator.astype(float)

    measure_table.numerator = measure_table.numerator.apply(
        lambda x: round_values(x, round_to, redact=redact, redaction_threshold=redaction_threshold)
    )
    measure_table.denominator = measure_table.denominator.apply(
        lambda x: round_values(x, round_to, redact=redact, redaction_threshold=redaction_threshold)
    )
    # recompute value
    measure_table.value = measure_table.numerator / measure_table.denominator
    return measure_table


def _redacted_string(measure_table):
    """
    Replace redacted values with "[REDACTED]" string
    A group could have the name NaN, so apply to specific columns
    """
    REDACTED_STR = "[REDACTED]"
    measure_table.numerator = measure_table.numerator.replace(
        numpy.nan, REDACTED_STR
    )
    measure_table.denominator = measure_table.denominator.replace(
        numpy.nan, REDACTED_STR
    )
    measure_table.value = measure_table.value.replace(numpy.nan, REDACTED_STR)
    return measure_table


def write_table(measure_table, path, filename):
    create_dir(path)
    measure_table.to_csv(path / filename, index=False, header=True)


def create_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def get_path(*args):
    return pathlib.Path(*args).resolve()


def match_input(input_list):
    path = get_path(input_list)
    if path.exists():
        return path


def match_paths(pattern):
    return [get_path(x) for x in glob.glob(pattern)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-files",
        required=True,
        type=match_paths,
        action="append",
        help="Glob pattern(s) for matching one or more input files",
    )
    parser.add_argument(
        "--exclude-files",
        required=False,
        type=match_paths,
        action="append",
        default=[],
        help="Glob pattern(s) to exclude one or more input files",
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
        help="Name for joined measures file",
    )
    parser.add_argument(
        "--round-to",
        required=False,
        default=10,
        type=int,
        help="Round to the nearest",
    )
    parser.add_argument(
        "--skip-round",
        action="store_true",
        help="Just join, do not round",
    )
    parser.add_argument(
        "--allow-practice",
        action="store_true",
        help="Allow practice-level data in joined measures file",
    )
    parser.add_argument(
        "--redact",
        action="store_true",
        help="Redact values below a threshold",
    )
    parser.add_argument(
        "--redaction-threshold",
        required=False,
        default=5,
        type=int,
        help="Redact values below or equal to this threshold",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_files = args.input_files
    exclude_files = args.exclude_files
    output_dir = args.output_dir
    output_name = args.output_name
    round_to = args.round_to
    skip_round = args.skip_round
    allow_practice = args.allow_practice
    redact = args.redact
    redaction_threshold = args.redaction_threshold

    tables = []
    for measure_table in get_measure_tables(input_files, exclude_files):
        table = _reshape_data(measure_table)
        names_ensured = _ensure_names(table)
        if not skip_round:
            names_ensured = _round_table(names_ensured, round_to, redact, redaction_threshold)
        redacted_str = _redacted_string(names_ensured)
        tables.append(redacted_str)

    output = _join_tables(tables)
    if not allow_practice:
        _check_for_practice(output)

    write_table(output, output_dir, output_name)


if __name__ == "__main__":
    main()
