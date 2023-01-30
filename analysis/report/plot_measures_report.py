import pandas as pd
import argparse
import pathlib
from report_utils import (
    plot_measures,
    coerce_numeric,
    MEDICATION_TO_CODELIST,
    CLINICAL_TO_CODELIST,
)


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

    df = pd.read_csv(measure_path, parse_dates=["date"])
    df = coerce_numeric(df)
    df["rate"] = 1000 * df["value"]

    # Stacked bar chart of population measures
    # TODO: name has extra text i.e. event_*_rate
    population_measures = df[df.group == "population"]
    # Medications
    medication_measures = [
        f"event_{x}_rate" for x in list(MEDICATION_TO_CODELIST.keys())
    ]
    medications = population_measures[
        population_measures.name.str.contains(
            "|".join(medication_measures)
        )
    ]
    plot_measures(
        medications,
        filename=output_dir / "medications_bar_measures_count",
        column_to_plot="numerator",
        y_label="Count of patients",
        as_bar=True,
        category="name",
    )
    plot_measures(
        medications,
        filename=output_dir / "medications_bar_measures",
        column_to_plot="rate",
        y_label="Rate per 1000",
        as_bar=True,
        category="name",
    )
    # Clinical
    clinical_measures = [
        f"event_{x}_rate" for x in list(CLINICAL_TO_CODELIST.keys())
    ]
    clinical = population_measures[
        population_measures.name.str.contains("|".join(clinical_measures))
    ]
    plot_measures(
        clinical,
        filename=output_dir / "clinical_bar_measures_count",
        column_to_plot="numerator",
        y_label="Count of patients",
        as_bar=True,
        category="name",
    )
    plot_measures(
        clinical,
        filename=output_dir / "clinical_bar_measures",
        column_to_plot="rate",
        y_label="Rate per 1000",
        as_bar=True,
        category="name",
    )

    # for each group, plot the measure
    for group, df_subset in df.groupby("name"):
        if "practice" in group:
            continue
        df_subset["rate"] = df_subset["value"] * 1000

        if len(df_subset["group"].unique()) == 1:
            # no breakdown

            plot_measures(
                df_subset,
                filename=output_dir / f"plot_measures_{group}",
                column_to_plot="rate",
                y_label="Rate per 1000",
                as_bar=False,
                category=None,
            )
        else:
            plot_measures(
                df_subset,
                filename=output_dir / f"plot_measures_{group}",
                column_to_plot="rate",
                y_label="Rate per 1000",
                as_bar=False,
                category="group",
            )


if __name__ == "__main__":
    main()
