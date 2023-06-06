import argparse
import pathlib
import pandas

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
        "--practice-file",
        required=False,
        help="Path to extra practice file",
    )
    parser.add_argument(
        "--panels",
        required=True,
        nargs="+",
        help="Panel subgroups to plot in order",
    )
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
        "--xtick-frequency",
        help="Display every nth xtick",
        type=int,
        default=1,
    )
    return parser.parse_args()


def get_measures_list(key, panels):
    measures_list = []
    for panel in panels:
        if panel == "population":
            measures_list.append(f"event_{key}_rate")
        else:
            measures_list.append(f"event_{key}_{panel}_rate")
    return measures_list


def panels_with(measure_table, panels, output_dir, frequency, xtick_frequency):
    pairs = {x: "clinical_any" for x in MEDICATION_TO_CODELIST.keys()}
    pairs.update({x: "medication_any" for x in CLINICAL_TO_CODELIST.keys()})
    for event, with_key in pairs.items():
        prefix = f"{event}_with_{with_key}"
        column_to_plot = "value"
        measure_list = get_measures_list(prefix, panels)
        subset = subset_table(measure_table, None, measure_list)
        if subset.empty:
            return
        # NOTE: the denominator of these measures is not the population, use %
        chart, lgds = get_group_chart(
            subset,
            panels,
            column_to_plot=column_to_plot,
            columns=2,
            date_lines=None,
            scale="rate",
            ci=None,
            exclude_group="Missing",
            output_dir=output_dir,
            frequency=frequency,
            xtick_frequency=xtick_frequency,
        )
        output_name = f"{prefix}_by_subgroup"
        plot_title = filename_to_title(output_name)
        write_group_chart(chart, lgds, output_dir / output_name, plot_title)
        chart.close()


def panels_loop(measure_table, panels, output_dir, frequency, xtick_frequency):
    for key in (
        list(MEDICATION_TO_CODELIST.keys())
        + ["medication_any", "clinical_any"]
        + list(CLINICAL_TO_CODELIST.keys())
    ):
        # Rate
        column_to_plot = "value"
        measure_list = get_measures_list(key, panels)
        subset = subset_table(measure_table, None, measure_list)
        chart, lgds = get_group_chart(
            subset,
            panels,
            column_to_plot=column_to_plot,
            columns=2,
            date_lines=None,
            scale="rate",
            ci=None,
            exclude_group="Missing",
            output_dir=output_dir,
            frequency=frequency,
            xtick_frequency=xtick_frequency,
        )
        output_name = f"{key}_by_subgroup"
        plot_title = filename_to_title(output_name)
        write_group_chart(chart, lgds, output_dir / output_name, plot_title)
        chart.close()

        # Count
        column_to_plot = "numerator"
        measure_list = get_measures_list(key, panels)
        subset = subset_table(measure_table, None, measure_list)
        chart_count, lgds_count = get_group_chart(
            subset,
            panels,
            column_to_plot=column_to_plot,
            columns=2,
            date_lines=None,
            scale=None,
            ci=None,
            exclude_group="Missing",
            output_dir=output_dir,
            frequency=frequency,
            xtick_frequency=xtick_frequency,
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
    practice_file = args.practice_file
    panels = args.panels
    output_dir = args.output_dir
    frequency = args.frequency
    xtick_frequency = args.xtick_frequency

    output_dir.mkdir(parents=True, exist_ok=True)

    measure_table = get_measure_tables(input_file)
    if practice_file:
        practice_table = get_measure_tables(practice_file)
        practice_table = practice_table[practice_table.category == "practice"]
        measure_table = pandas.concat([measure_table, practice_table])

    panels_loop(
        measure_table,
        panels,
        output_dir,
        frequency=frequency,
        xtick_frequency=xtick_frequency,
    )
    panels_with(
        measure_table,
        panels,
        output_dir,
        frequency=frequency,
        xtick_frequency=xtick_frequency,
    )


if __name__ == "__main__":
    main()
