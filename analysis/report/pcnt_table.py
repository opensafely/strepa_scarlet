import pathlib
import argparse
import pandas
import fnmatch

from report_utils import ci_95_proportion, ci_to_str


def get_measure_tables(input_file):
    measure_table = pandas.read_csv(
        input_file,
        dtype={"numerator": float, "denominator": float, "value": float},
        na_values="[REDACTED]",
    )

    return measure_table


def subset_table(measure_table, measures_pattern, date):
    """
    Given either a pattern of list of names, extract the subset of a joined
    measures file based on the 'name' column
    """

    measure_table = measure_table[measure_table["date"] == date]

    measures_list = []
    for pattern in measures_pattern:
        paths_to_add = match_paths(measure_table["name"], pattern)
        if len(paths_to_add) == 0:
            raise ValueError(f"Pattern did not match any rows: {pattern}")
        measures_list += paths_to_add

    table_subset = measure_table[measure_table["name"].isin(measures_list)]

    if table_subset.empty:
        raise ValueError("Patterns did not match any rows")

    return table_subset


def match_paths(files, pattern):
    return fnmatch.filter(files, pattern)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to single joined measures file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="Path to the output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input_file
    output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    measure_table = get_measure_tables(input_file)

    end_date = measure_table.date.max()
    medications = subset_table(
        measure_table, ["event_*_with_clinical_any_pcnt"], end_date
    )
    medications.index = pandas.MultiIndex.from_product(
        [
            ["Medication with clinical any"],
            list(
                medications.name.str.extract(
                    r"event_(\w+)_with_clinical_any_pcnt", expand=False
                ).str.title()
            ),
        ]
    )
    medications = medications.sort_values("value", ascending=False)
    medications_cis = ci_to_str(
        ci_95_proportion(medications, scale=100), decimals=1
    )
    clinical = subset_table(
        measure_table, ["event_*_with_medication_any_pcnt"], end_date
    )
    clinical.index = pandas.MultiIndex.from_product(
        [
            ["Clinical with medication any"],
            list(
                clinical.name.str.extract(
                    r"event_(\w+)_with_medication_any_pcnt", expand=False
                ).str.title()
            ),
        ]
    )
    clinical = clinical.sort_values("value", ascending=False)
    clinical_cis = ci_to_str(ci_95_proportion(clinical, scale=100), decimals=1)
    joined = pandas.concat([medications_cis, clinical_cis])
    joined.name = "Percent (95% CI)"
    table = pandas.DataFrame(joined)
    table.to_html(output_dir / "pcnt_with_indication.html", index=True)


if __name__ == "__main__":
    main()
