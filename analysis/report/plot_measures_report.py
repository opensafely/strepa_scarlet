import pandas as pd
import numpy as np
import argparse
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
from report_utils import (
    add_date_lines,
    autoselect_labels,
    colour_palette,
    get_measure_tables,
    translate_group,
    parse_date,
    set_fontsize,
    make_season_table,
    annotate_seasons,
    MEDICATION_TO_CODELIST,
    CLINICAL_TO_CODELIST,
)

import matplotlib.ticker as ticker

ticker.Locator.MAXTICKS = 10000


def plot_measures(
    df,
    output_dir: pathlib.Path,
    filename: str,
    column_to_plot: str,
    y_label: str,
    as_bar: bool = False,
    category: str = None,
    frequency: str = "month",
    log_scale: bool = False,
    legend_inside: bool = False,
    mark_seasons: bool = False,
    produce_season_table: bool = False,
    date_lines: list = [],
):
    """Produce time series plot from measures table. One line is plotted for
    each sub category within the category column. Saves output as jpeg file.
    Args:
        df: A measure table
        column_to_plot: Column name for y-axis values
        y_label: Label to use for y-axis
        as_bar: Boolean indicating if bar chart should be plotted instead of
                line chart.
        category: Name of column indicating different categories
    """
    df_copy = df.copy()

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(18, 10))
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colour_palette)
    if log_scale:
        plt.yscale("log")

    # NOTE: finite filter for dummy data
    y_max = (
        df_copy[np.isfinite(df_copy[column_to_plot])][column_to_plot].max()
        * 1.50
    )
    # Ignore timestamp - this could be done at load time
    df_copy["date"] = df_copy["date"].dt.date

    df_copy = df_copy.set_index("date")
    if category:
        # Set up category to have clean labels
        repeated = autoselect_labels(df_copy[category])
        df_copy[category] = df_copy.apply(
            lambda x: translate_group(
                x[category], x[category], repeated, True
            ),
            axis=1,
        )
        if as_bar:
            df_copy.pivot(columns=category, values=column_to_plot).plot.bar(
                ax=ax, stacked=True
            )
            y_max = (
                df_copy.groupby(["date"])[column_to_plot].sum().max() * 1.05
            )
        else:
            df_copy.groupby(category)[column_to_plot].plot(ax=ax)
    else:
        if as_bar:
            df_copy[column_to_plot].bar(ax=ax, legend=False)
        else:
            df_copy[column_to_plot].plot(ax=ax, legend=False)

    if frequency == "month":
        xticks = pd.date_range(
            start=df_copy.index.min(), end=df_copy.index.max(), freq="MS"
        )
        ax.set_xticks(xticks)
        ax.set_xticklabels([x.strftime("%B %Y") for x in xticks])

        # TODO: check that category exists?
        if produce_season_table or mark_seasons:
            season_table = make_season_table(
                df_copy, category, column_to_plot, output_dir, filename
            )
            # TODO: check whether this works for weekly
            if not season_table.empty and mark_seasons:
                annotate_seasons(season_table, column_to_plot, ax)

    elif frequency == "week":
        xticks = pd.date_range(
            start=df_copy.index.min(), end=df_copy.index.max(), freq="W-THU"
        )
        ax.set_xticks(xticks)
        ax.set_xticklabels([x.strftime("%d-%m-%Y") for x in xticks])

    if log_scale:
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        y_label = f"{y_label} (displayed on log scale)"

    plt.ylabel(y_label)
    plt.xlabel("Date")
    plt.xticks(rotation="vertical")
    plt.xticks(fontsize="medium")

    plt.ylim(
        top=1000 if df_copy[column_to_plot].isnull().all() else y_max,
    )

    if category:
        if legend_inside:
            plt.legend(
                sorted(df_copy[category].unique()),
                ncol=3,
            )
        else:
            plt.legend(
                sorted(df_copy[category].unique()),
                bbox_to_anchor=(1.04, 1),
                loc="upper left",
            )
    if date_lines:
        min_date = min(df_copy.index)
        max_date = max(df_copy.index)
        add_date_lines(date_lines, min_date, max_date)
    plt.tight_layout()

    plt.savefig(output_dir / f"{filename}.jpeg")
    plt.close()


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
        "--produce-season-table",
        action="store_true",
        help="Generate a table with the max and min of each season",
    )
    parser.add_argument(
        "--date-lines",
        nargs="+",
        type=parse_date,
        help="Vertical date lines",
    )
    parser.add_argument(
        "--base-fontsize",
        help="Default text size",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    measure_path = args.measure_path
    output_dir = args.output_dir
    frequency = args.frequency
    log_scale = args.log_scale
    legend_inside = args.legend_inside
    mark_seasons = args.mark_seasons
    produce_season_table = args.produce_season_table
    date_lines = args.date_lines
    base_fontsize = args.base_fontsize

    output_dir.mkdir(parents=True, exist_ok=True)
    set_fontsize(base_fontsize)

    df = get_measure_tables(measure_path)
    df["rate"] = 1000 * df["value"]

    population_measures = df[df.group == "population"]

    # Medications
    medication_measures = [
        f"event_{x}_rate" for x in list(MEDICATION_TO_CODELIST.keys())
    ]
    medications = population_measures[
        population_measures.name.str.contains("|".join(medication_measures))
    ]
    # NOTE: too many medications to annotate counts on plot
    plot_measures(
        medications,
        output_dir=output_dir,
        filename="medications_bar_measures_count",
        column_to_plot="numerator",
        y_label="Count of patients",
        as_bar=False,
        category="name",
        frequency=frequency,
        log_scale=log_scale,
        legend_inside=legend_inside,
        mark_seasons=False,
        produce_season_table=produce_season_table,
        date_lines=date_lines,
    )
    # NOTE: too many medications to annotate counts on plot
    plot_measures(
        medications,
        output_dir=output_dir,
        filename="medications_bar_measures",
        column_to_plot="rate",
        y_label="Rate per 1000 patients",
        as_bar=False,
        category="name",
        frequency=frequency,
        log_scale=log_scale,
        legend_inside=legend_inside,
        mark_seasons=False,
        produce_season_table=produce_season_table,
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
        output_dir=output_dir,
        filename="clinical_bar_measures_count",
        column_to_plot="numerator",
        y_label="Count of patients",
        as_bar=False,
        category="name",
        frequency=frequency,
        log_scale=log_scale,
        legend_inside=legend_inside,
        mark_seasons=mark_seasons,
        produce_season_table=produce_season_table,
        date_lines=date_lines,
    )
    plot_measures(
        clinical,
        output_dir=output_dir,
        filename="clinical_bar_measures",
        column_to_plot="rate",
        y_label="Rate per 1000 patients",
        as_bar=False,
        category="name",
        frequency=frequency,
        log_scale=log_scale,
        legend_inside=legend_inside,
        mark_seasons=mark_seasons,
        produce_season_table=produce_season_table,
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
        # NOTE: too many medications to annotate counts on plot
        plot_measures(
            medications_with_clinical,
            output_dir=output_dir,
            filename="medications_with_clinical_bar_measures_count",
            column_to_plot="numerator",
            y_label="Count of patients",
            as_bar=False,
            category="name",
            frequency=frequency,
            log_scale=log_scale,
            legend_inside=legend_inside,
            mark_seasons=False,
            produce_season_table=produce_season_table,
            date_lines=date_lines,
        )
        # NOTE: too many medications to annotate counts on plot
        plot_measures(
            medications_with_clinical,
            output_dir=output_dir,
            filename="medications_with_clinical_bar_measures",
            column_to_plot="rate",
            y_label="Rate per 1000",
            as_bar=False,
            category="name",
            frequency=frequency,
            log_scale=log_scale,
            legend_inside=legend_inside,
            mark_seasons=False,
            produce_season_table=produce_season_table,
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
            output_dir=output_dir,
            filename="clinical_with_medication_bar_measures_count",
            column_to_plot="numerator",
            y_label="Count of patients",
            as_bar=False,
            category="name",
            frequency=frequency,
            log_scale=log_scale,
            legend_inside=legend_inside,
            mark_seasons=mark_seasons,
            produce_season_table=produce_season_table,
            date_lines=date_lines,
        )
        plot_measures(
            clinical_with_medication,
            output_dir=output_dir,
            filename="clinical_with_medication_bar_measures",
            column_to_plot="rate",
            y_label="Rate per 1000",
            as_bar=False,
            category="name",
            frequency=frequency,
            log_scale=log_scale,
            legend_inside=legend_inside,
            mark_seasons=mark_seasons,
            produce_season_table=produce_season_table,
            date_lines=date_lines,
        )


if __name__ == "__main__":
    main()
