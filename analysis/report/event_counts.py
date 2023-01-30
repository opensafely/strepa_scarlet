# load each input file and count the number of unique patients and number of events
from pathlib import Path
from report_utils import match_input_files, get_date_input_file, save_to_json
import pandas as pd
import argparse
import numpy as np


def round_to_nearest_100(x):
    return int(round(x, -2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--measures", type=str, required=True)
    return parser.parse_args()


def get_column_uniques(df, column):
    return df.loc[:, column].unique()


def get_column_sum(df, column):
    return df.loc[:, column].sum()


def main():
    args = parse_args()
    measures = args.measures.split(",")

    for measure in measures:
        patients = []
        events = {}

        for file in args.input_dir.iterdir():
            if match_input_files(file.name):
                date = get_date_input_file(file.name)
                df = pd.read_csv(file)
                df["date"] = date
                num_events = get_column_sum(df, f"event_{measure}")
                events[date] = num_events
                unique_patients = get_column_uniques(df.loc[df[f"event_{measure}"]==1,:], "patient_id")
                patients.extend(unique_patients)

        total_events = round_to_nearest_100(sum(events.values()))
        total_patients = round_to_nearest_100(len(np.unique(patients)))
        events_in_latest_period = round_to_nearest_100(
            events[max(events.keys())]
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_to_json(
            {
                "total_events": total_events,
                "total_patients": total_patients,
                "events_in_latest_period": events_in_latest_period,
            },
            args.output_dir / f"event_counts_{measure}.json",
        )


if __name__ == "__main__":
    main()
