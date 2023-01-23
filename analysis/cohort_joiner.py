import argparse
import glob
import pathlib

import pandas
from cohortextractor import pandas_utils


def read_dataframe(path):
    ext = get_extension(path)
    if ext == ".csv" or ext == ".csv.gz":
        return pandas.read_csv(path, dtype="str")
    elif ext == ".feather":
        return pandas.read_feather(path)
    elif ext == ".dta" or ext == ".dta.gz":
        return pandas.read_stata(path)
    else:
        raise ValueError(f"Cannot read '{ext}' files")


def write_dataframe(dataframe, path):
    # We refactored this function, replacing copy-and-paste code from cohort-extractor
    # with a call to cohort-extractor itself. However, the error types differed. We
    # preserved the pre-refactoring error type.
    ext = get_extension(path)
    if ext == ".feather":
        # We write feather files ourselves, because of an issue with dependencies.
        # For more information, see:
        # https://github.com/opensafely-actions/cohort-joiner/issues/25
        dataframe.to_feather(path)
    else:
        try:
            pandas_utils.dataframe_to_file(dataframe, path)
        except RuntimeError:
            raise ValueError(f"Cannot write '{ext}' files")


def get_extension(path):
    return "".join(path.suffixes)


def left_join(lhs_dataframe, rhs_dataframe):
    return lhs_dataframe.merge(rhs_dataframe, how="left", on="patient_id")


def get_path(*args):
    if is_glob("/".join(args)):
        raise argparse.ArgumentTypeError(
            "Argument should be a path and not a glob pattern"
        )
    else:
        return pathlib.Path(*args).resolve()


def match_paths(pattern):
    return [get_path(x) for x in glob.glob(pattern)]


def is_glob(s):
    return any(c in ["*", "?", "[", "]"] for c in s)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lhs",
        dest="lhs_paths",
        required=True,
        type=match_paths,
        metavar="LHS_PATTERN",
        help="Glob pattern for matching one or more input dataframes that will form the left-hand side of the join",
    )
    parser.add_argument(
        "--rhs",
        dest="rhs_path",
        required=True,
        type=get_path,
        metavar="RHS_FILE",
        help="The input dataframe that will form the right-hand side of the join",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_path",
        default="output/joined",
        type=get_path,
        metavar="OUTPUT_DIR",
        help="The output directory. If it doesn't exist, then it will be created",
    )
    return parser.parse_args()


def check_paths(lhs_paths, rhs_path, output_path):
    if any(output_path == path.parent for path in lhs_paths + [rhs_path]):
        raise ValueError(
            "The output directory cannot contain the input dataframes"
        )


def main():
    args = parse_args()
    lhs_paths = args.lhs_paths
    rhs_path = args.rhs_path
    output_path = args.output_path
    check_paths(lhs_paths, rhs_path, output_path)

    rhs_dataframe = read_dataframe(rhs_path)
    for lhs_path in lhs_paths:
        lhs_dataframe = read_dataframe(lhs_path)
        lhs_dataframe = left_join(lhs_dataframe, rhs_dataframe)
        # We only make output_path when there's a lhs_dataframe to write to it
        output_path.mkdir(parents=True, exist_ok=True)
        write_dataframe(lhs_dataframe, output_path / lhs_path.name)


if __name__ == "__main__":
    main()
