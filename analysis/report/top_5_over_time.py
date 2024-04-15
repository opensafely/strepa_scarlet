import argparse
import pandas as pd
from pathlib import Path

from top_5_report import plot_top_codes_over_time


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
        type=Path,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--frequency",
        default="month",
        choices=["month", "week"],
        help="The frequency of the data",
    )
    parser.add_argument(
        "--xtick-frequency",
        help="Display every nth xtick",
        type=int,
        default=1,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input_file)
    plot_top_codes_over_time(df, "sore_throat_tonsillitis_rate", args.output_dir, args.frequency, args.xtick_frequency)

if __name__ == "__main__":
    main()
