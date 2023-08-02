import pandas as pd
import argparse
import pathlib
from report_utils import get_measure_tables, GROUPED_MEDICATIONS


def group_medications(medications_df):
    by_line = []
    for line, meds in GROUPED_MEDICATIONS.items():
        criteria = "(" + "|".join(meds) + ")"
        medications = medications_df[
            medications_df.name.str.contains("|".join(meds))
        ]
        medications.name = medications.name.str.replace(
            criteria, line, regex=True
        )
        medications = medications.groupby(
            ["date", "category", "group", "name"], as_index=False
        ).agg({"numerator": "sum", "denominator": lambda x: x.iloc[0]})
        medications["value"] = (
            medications["numerator"] / medications["denominator"]
        )
        medications["rate"] = 1000 * medications["value"]
        by_line.append(medications)
    return pd.concat(by_line)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure-path", help="path of combined measure file")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="Path to the output directory",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    measure_path = args.measure_path
    output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    df = get_measure_tables(measure_path)
    grouped = group_medications(df)
    grouped.to_csv(output_dir / "grouped_measures.csv", index=False)


if __name__ == "__main__":
    main()
