import argparse
import pathlib
import re
import glob
import pandas
import itertools
from report.report_utils import round_values

FNAME_REGEX = re.compile(r"input_excluded_(?P<id>\S+)\.csv.gz")


def _round_table(table, round_to, redact=False, redaction_threshold=5):
    table = table.astype(float)

    table = table.apply(
        lambda x: round_values(
            x, round_to, redact=redact, redaction_threshold=redaction_threshold
        )
    )
    table = table.fillna("[REDACTED]")
    return table


def _join_tables(tables):
    return pandas.concat(tables)


def get_input_tables(input_files, exclude_files):
    all_files = set(itertools.chain(*input_files))
    all_exclude = set(itertools.chain(*exclude_files))
    all_files = all_files - all_exclude
    for input_file in all_files:
        measure_fname_match = re.match(FNAME_REGEX, input_file.name)
        if measure_fname_match is not None:
            # The `date` column is assigned by the measures framework.
            measure_table = pandas.read_csv(input_file)

            # We can reconstruct the parameters passed to `Measure` without
            # the study definition.
            measure_table.attrs["id"] = measure_fname_match.group("id")
            yield measure_table


def compute_excluded(input_table):
    d = {}
    d["total"] = len(input_table)
    registered = input_table[input_table.registered == 1]
    d["not_registered"] = len(input_table) - len(registered)
    alive = registered[registered.died == 0]
    d["died"] = len(registered) - len(alive)
    age = alive[alive.age != "missing"]
    d["unknown_age"] = len(alive) - len(age)
    sex = age[(age.sex == "M") | (age.sex == "F")]
    d["unknown_sex"] = len(age) - len(sex)
    excluded = input_table[input_table.included == 0]
    d["total_excluded"] = len(excluded)
    d["clinical_any"] = (excluded.event_clinical_any == 1).sum()
    d["medication_any"] = (excluded.event_medication_any == 1).sum()
    counts = pandas.Series(d)
    counts.name = "count"
    counts.index.name = "attribute"
    return counts


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
    redact = args.redact
    redaction_threshold = args.redaction_threshold

    tables = []
    for input_table in get_input_tables(input_files, exclude_files):
        table_date = input_table.attrs["id"]
        excluded_counts = compute_excluded(input_table)
        redacted_and_rounded = _round_table(
            excluded_counts, round_to, redact, redaction_threshold
        )
        df = redacted_and_rounded.reset_index()
        df["date"] = table_date
        tables.append(df)

    output = _join_tables(tables)

    write_table(output, output_dir, output_name)


if __name__ == "__main__":
    main()
