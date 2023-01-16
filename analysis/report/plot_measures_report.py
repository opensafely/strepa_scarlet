import pandas as pd
import argparse
import pathlib
from report_utils import plot_measures, coerce_numeric


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
    # Might have to filter for the medication measures
    # TODO: name has extra text i.e. event_*_rate
    population_measures = df[df.group == "population"]
    plot_measures(
        population_measures,
        filename=output_dir / "bar_measures",
        column_to_plot="rate",
        y_label="Rate per 1000",
        as_bar=True,
        category="name",
    )

    # for each group, plot the measure
    for group, df_subset in df.groupby("name"):
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
