import pandas as pd
import argparse
import pathlib
from report_utils import (
    parse_date,
    plot_measures,
    coerce_numeric,
    MEDICATION_TO_CODELIST,
    CLINICAL_TO_CODELIST,
    GROUPED_MEDICATIONS,
)


def group_medications(medications_df):
    by_line = []
    for line, meds in GROUPED_MEDICATIONS.items():
        medications = (
            medications_df[medications_df.name.str.contains("|".join(meds))]
            .groupby("date", as_index=False)
            .agg({"numerator": "sum", "denominator": lambda x: x.iloc[0]})
        )
        medications["name"] = line
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
    parser.add_argument(
        "--frequency",
        default="month",
        choices=["month", "week"],
        help="Frequency of data",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Display y axis on log scale",
    )
    parser.add_argument(
        "--use-groups",
        action="store_true",
        help="Group medications into first, second, and third line",
    )
    parser.add_argument(
        "--legend-inside",
        action="store_true",
        help="Place the legend inside the plot",
    )
    parser.add_argument(
        "--mark-seasons",
        action="store_true",
        help="Mark the max and min of each season",
    )
    parser.add_argument(
        "--date-lines",
        nargs="+",
        type=parse_date,
        help="Vertical date lines",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    measure_path = args.measure_path
    output_dir = args.output_dir
    frequency = args.frequency
    log_scale = args.log_scale
    use_groups = args.use_groups
    legend_inside = args.legend_inside
    mark_seasons = args.mark_seasons
    date_lines = args.date_lines

    df = pd.read_csv(measure_path, parse_dates=["date"])
    df = coerce_numeric(df)
    df["rate"] = 1000 * df["value"]

    population_measures = df[df.group == "population"]

    # Medications
    medication_measures = [
        f"event_{x}_rate" for x in list(MEDICATION_TO_CODELIST.keys())
    ]
    medications = population_measures[
        population_measures.name.str.contains("|".join(medication_measures))
    ]
    if use_groups:
        medications = group_medications(medications)

    plot_measures(
        medications,
        filename=output_dir / "medications_bar_measures_count",
        column_to_plot="numerator",
        y_label="Count of patients",
        as_bar=False,
        category="name",
        frequency=frequency,
        log_scale=log_scale,
        legend_inside=legend_inside,
        mark_seasons=mark_seasons,
        date_lines=date_lines,
    )
    plot_measures(
        medications,
        filename=output_dir / "medications_bar_measures",
        column_to_plot="rate",
        y_label="Rate per 1000 patients",
        as_bar=False,
        category="name",
        frequency=frequency,
        log_scale=log_scale,
        legend_inside=legend_inside,
        mark_seasons=mark_seasons,
        date_lines=date_lines,
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
        as_bar=False,
        category="name",
        frequency=frequency,
        log_scale=log_scale,
        legend_inside=legend_inside,
        mark_seasons=mark_seasons,
        date_lines=date_lines,
    )
    plot_measures(
        clinical,
        filename=output_dir / "clinical_bar_measures",
        column_to_plot="rate",
        y_label="Rate per 1000 patients",
        as_bar=False,
        category="name",
        frequency=frequency,
        log_scale=log_scale,
        legend_inside=legend_inside,
        mark_seasons=mark_seasons,
        date_lines=date_lines,
    )

    # Only the monthly data has "with" measures
    if frequency == "month":
        # Medications with clinical
        medication_with_clinical_measures = [
            f"event_{x}_with_clinical_any_rate"
            for x in list(MEDICATION_TO_CODELIST.keys())
        ]
        medications_with_clinical = population_measures[
            population_measures.name.str.contains(
                "|".join(medication_with_clinical_measures)
            )
        ]
        if use_groups:
            medications_with_clinical = group_medications(
                medications_with_clinical
            )
        plot_measures(
            medications_with_clinical,
            filename=output_dir
            / "medications_with_clinical_bar_measures_count",
            column_to_plot="numerator",
            y_label="Count of patients",
            as_bar=False,
            category="name",
            frequency=frequency,
            log_scale=log_scale,
            legend_inside=legend_inside,
            mark_seasons=mark_seasons,
            date_lines=date_lines,
        )
        plot_measures(
            medications_with_clinical,
            filename=output_dir / "medications_with_clinical_bar_measures",
            column_to_plot="rate",
            y_label="Rate per 1000",
            as_bar=False,
            category="name",
            frequency=frequency,
            log_scale=log_scale,
            legend_inside=legend_inside,
            mark_seasons=mark_seasons,
            date_lines=date_lines,
        )

        # Clinical with medication
        clinical_with_medication_measures = [
            f"event_{x}_with_medication_any_rate"
            for x in list(CLINICAL_TO_CODELIST.keys())
        ]
        clinical_with_medication = population_measures[
            population_measures.name.str.contains(
                "|".join(clinical_with_medication_measures)
            )
        ]
        plot_measures(
            clinical_with_medication,
            filename=output_dir
            / "clinical_with_medication_bar_measures_count",
            column_to_plot="numerator",
            y_label="Count of patients",
            as_bar=False,
            category="name",
            frequency=frequency,
            log_scale=log_scale,
            legend_inside=legend_inside,
            mark_seasons=mark_seasons,
            date_lines=date_lines,
        )
        plot_measures(
            clinical_with_medication,
            filename=output_dir / "clinical_with_medication_bar_measures",
            column_to_plot="rate",
            y_label="Rate per 1000",
            as_bar=False,
            category="name",
            frequency=frequency,
            log_scale=log_scale,
            legend_inside=legend_inside,
            mark_seasons=mark_seasons,
            date_lines=date_lines,
        )


if __name__ == "__main__":
    main()
