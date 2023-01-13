import pandas as pd
import argparse
from report_utils import plot_measures


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure_path", help="path of combined measure file")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    df = pd.read_csv(args.measure_path, parse_dates=["date"])
    df = df.loc[df["value"] != "[REDACTED]", :]

    # for each group, plot the measure
    for group in df["name"].unique():
        df_subset = df.loc[df["name"] == group, :]
        df_subset["value"] = df_subset["value"].astype(float)
        df_subset["rate"] = df_subset["value"] * 1000

        df_subset["group"] = df_subset["group"].astype(str)
        if len(df_subset["group"].unique()) == 1:
            # no breakdown

            plot_measures(
                df_subset,
                filename="report/plot_measures",
                column_to_plot="rate",
                y_label="Rate per 1000",
                as_bar=False,
                category=None,
            )
        else:
            plot_measures(
                df_subset,
                filename=f"report/plot_measures_{group}",
                column_to_plot="rate",
                y_label="Rate per 1000",
                as_bar=False,
                category="group",
            )


if __name__ == "__main__":
    main()
