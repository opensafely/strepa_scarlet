import pandas as pd
import numpy as np
import argparse
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
from report_utils import (
    ci_to_str,
    ci_95_proportion,
    add_date_lines,
    autoselect_labels,
    colour_palette,
    translate_group,
    parse_date,
    coerce_numeric,
    set_fontsize,
    MEDICATION_TO_CODELIST,
    CLINICAL_TO_CODELIST,
    GROUPED_MEDICATIONS,
)

import matplotlib.ticker as ticker

ticker.Locator.MAXTICKS = 10000


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


# NOTE: bug with pandas 1.01, cannot do pivot with MultiIndex
def MultiIndex_pivot(
    df: pd.DataFrame,
    index: str = None,
    columns: str = None,
    values: str = None,
) -> pd.DataFrame:
    """
    https://github.com/pandas-dev/pandas/issues/23955
    Usage:
    df.pipe(MultiIndex_pivot, index = ['idx_column1', 'idx_column2'],
            columns = ['col_column1', 'col_column2'], values = 'bar')
    """
    output_df = df.copy(deep=True)
    if index is None:
        names = list(output_df.index.names)
        output_df.reset_index(drop=True, inplace=True)
    else:
        names = index
    output_df = output_df.assign(
        tuples_index=[tuple(i) for i in output_df[names].values]
    )
    if isinstance(columns, list):
        output_df = output_df.assign(
            tuples_columns=[tuple(i) for i in output_df[columns].values]
        )
        output_df = output_df.pivot(
            index="tuples_index", columns="tuples_columns", values=values
        )
        output_df.columns = pd.MultiIndex.from_tuples(
            [((x[0],) + x[1]) for x in output_df.columns],
            names=[None] + columns,
        )
    else:
        output_df = output_df.pivot(
            index="tuples_index", columns=columns, values=values
        )
    output_df.index = pd.MultiIndex.from_tuples(output_df.index, names=names)
    return output_df


def produce_min_max_table(df, column_to_plot, category):
    df.numerator = df.numerator.astype(float)
    df.denominator = df.denominator.astype(float)
    cis = ci_95_proportion(df, scale=1000)
    df["Rate (95% CI)"] = ci_to_str(cis)
    df = df.rename({"numerator": "Count"}, axis=1)
    table = df.pipe(
        MultiIndex_pivot,
        index=[category, "type"],
        columns="gas_year",
        values=["Count", "rate", "Rate (95% CI)"],
    )
    if column_to_plot == "numerator":
        table["Count"] = table["Count"].applymap(lambda x: f"{x:.0f}")
        return table["Count"]
    else:
        table.Count = table.Count.astype(float)
        table.rate = table.rate.astype(float)
        ratio = table["rate"]["2023"] / table["rate"]["2018"]

        # See formula of ci irr:
        # https://researchonline.lshtm.ac.uk/id/eprint/251164/1/pmed.1001270.s005.pdf
        sd_log_irr = np.sqrt(
            (1 / table["Count"]["2023"]) + (1 / table["Count"]["2018"])
        )
        lci = np.exp(np.log(ratio) - 1.96 * sd_log_irr)
        uci = np.exp(np.log(ratio) + 1.96 * sd_log_irr)
        rr_cis = pd.concat([ratio, lci, uci], axis=1)

        rr_cis_str = ci_to_str(rr_cis)
        rr_cis_str.name = ("2023 v 2018", "Rate Ratio (95% CI)")

        # Create table with count, rate (95% CI)
        table["Count"] = table["Count"].applymap(lambda x: f"{x:.0f}")
        count_cis = (table[["Count", "Rate (95% CI)"]]).swaplevel(axis=1)
        cols = count_cis.columns.get_level_values(0).unique()
        reordered = count_cis.reindex(columns=cols, level="gas_year")
        return pd.concat([reordered, rr_cis_str], axis=1)


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

        if produce_season_table or mark_seasons:
            # TODO: check whether this works for weekly
            df_copy["gas_year"] = pd.to_datetime(df_copy.index).to_period(
                "A-Aug"
            )
            mins_and_maxes = []
            for year, data in df_copy.groupby(["gas_year", category]):
                data_sorted = data.dropna(subset=[column_to_plot]).sort_values(
                    column_to_plot
                )
                if data_sorted.empty:
                    continue
                data_min = data_sorted.iloc[0]
                data_max = data_sorted.iloc[-1]
                if mark_seasons:
                    plt.annotate(
                        int(data_min[column_to_plot]),
                        xy=(data_min.name, data_min[column_to_plot]),
                        xytext=(0, 25),
                        textcoords="offset points",
                        verticalalignment="bottom",
                        arrowprops=dict(facecolor="black", shrink=0.025),
                        horizontalalignment="left"
                        if (data_min.name.month % 2) == 1
                        else "right",
                        fontsize=12,
                    )
                    plt.annotate(
                        int(data_max[column_to_plot]),
                        xy=(data_max.name, data_max[column_to_plot]),
                        xytext=(0, -25),
                        textcoords="offset points",
                        verticalalignment="top",
                        arrowprops=dict(facecolor="blue", shrink=0.025),
                        fontsize=12,
                    )
                data_min["type"] = "min"
                data_max["type"] = "max"
                mins_and_maxes.append(data_min)
                mins_and_maxes.append(data_max)
            if produce_season_table:
                table = produce_min_max_table(
                    pd.concat(mins_and_maxes, axis=1).T,
                    column_to_plot,
                    category,
                )
                table.to_html(output_dir / f"{filename}_table.html")

    elif frequency == "week":
        xticks = pd.date_range(
            start=df_copy.index.min(), end=df_copy.index.max(), freq="W-THU"
        )
        ax.set_xticks(xticks)
        ax.set_xticklabels([x.strftime("%d-%m-%Y") for x in xticks])

    if log_scale:
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        y_label = f"Log {y_label.lower()}"

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
    use_groups = args.use_groups
    legend_inside = args.legend_inside
    mark_seasons = args.mark_seasons
    produce_season_table = args.produce_season_table
    date_lines = args.date_lines
    base_fontsize = args.base_fontsize

    output_dir.mkdir(parents=True, exist_ok=True)
    set_fontsize(base_fontsize)

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

    if use_groups:
        medications = group_medications(medications)

        plot_measures(
            medications,
            output_dir=output_dir,
            filename="medications_grouped_bar_measures_count",
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
            medications,
            output_dir=output_dir,
            filename="medications_grouped_bar_measures",
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
        if use_groups:
            medications_with_clinical = group_medications(
                medications_with_clinical
            )
            plot_measures(
                medications_with_clinical,
                output_dir=output_dir,
                filename="medications_grouped_with_clinical_bar_measures_count",
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
                medications_with_clinical,
                output_dir=output_dir,
                filename="medications_grouped_with_clinical_bar_measures",
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
