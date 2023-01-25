import argparse
import pathlib

from report_utils import (
    filename_to_title,
    get_measure_tables,
    subset_table,
    write_group_chart,
    MEDICATION_TO_CODELIST,
    CLINICAL_TO_CODELIST,
)
from panel_plots import get_group_chart


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


def get_pattern_and_list(key, column_to_plot, first):
    measure_list = []
    if column_to_plot == "value":
        if key in MEDICATION_TO_CODELIST.keys():
            return (f"event_{key}*_rate", measure_list)
        else:
            demographics = [
                "age_band",
                "imd",
                "ethnicity",
                "region",
                "sex",
            ]
            return (
                None,
                [first] + [f"event_{key}_{d}_rate" for d in demographics],
            )
    else:
        return (
            None,
            [first]
            + [f"event_{key}_{d}_rate" for d in ["age_band", "region"]],
        )


def panels_loop(measure_table, output_dir):
    for key in (
        list(MEDICATION_TO_CODELIST.keys())
        + ["medication_any"]
        + list(CLINICAL_TO_CODELIST.keys())
    ):
        first = f"event_{key}_rate"
        # Rate
        column_to_plot = "value"
        pattern, measure_list = get_pattern_and_list(
            key, column_to_plot, first
        )
        subset = subset_table(measure_table, pattern, measure_list)
        chart, lgds = get_group_chart(
            subset,
            first,
            column_to_plot=column_to_plot,
            columns=2,
            date_lines=None,
            scale=None,
            ci=None,
            exclude_group="missing",
        )
        output_name = f"{key}_by_subgroup"
        plot_title = filename_to_title(output_name)
        write_group_chart(chart, lgds, output_dir / output_name, plot_title)
        chart.close()

        # Count
        column_to_plot = "numerator"
        pattern, measure_list = get_pattern_and_list(
            key, column_to_plot, first
        )
        subset = subset_table(measure_table, pattern, measure_list)
        chart_count, lgds_count = get_group_chart(
            subset,
            first,
            column_to_plot=column_to_plot,
            columns=2,
            date_lines=None,
            scale=None,
            ci=None,
            exclude_group="missing",
        )
        output_name_count = f"{key}_by_subgroup_count"
        plot_title_count = filename_to_title(output_name_count)
        write_group_chart(
            chart_count,
            lgds_count,
            output_dir / output_name_count,
            plot_title_count,
        )
        chart_count.close()


def main():
    args = parse_args()
    input_file = args.input_file
    output_dir = args.output_dir

    measure_table = get_measure_tables(input_file)
    panels_loop(measure_table, output_dir)


if __name__ == "__main__":
    main()
