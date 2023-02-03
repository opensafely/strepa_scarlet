import argparse
import pathlib
import pandas
import numpy
import re
import operator
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from dateutil import parser
from ebmdatalab import charts
from report_utils import (
    coerce_numeric,
    round_values,
    drop_zero_denominator_rows,
    filename_to_title,
    autoselect_labels,
    translate_group,
    get_measure_tables,
    subset_table,
    write_group_chart,
    colour_palette,
)


def scale_thousand(ax):
    """
    Scale a proportion for rate by 1000
    Used for y axis display
    """

    def thousand_formatter(x, pos):
        return f"{x*1000: .1f}"

    ax.yaxis.set_major_formatter(FuncFormatter(thousand_formatter))
    ax.set_ylabel("Rate per thousand")


def scale_hundred(ax):
    """
    Scale a proportion for percentage
    Used for y axis display
    """

    def hundred_formatter(x, pos):
        return f"{x*100: .0f}"

    ax.yaxis.set_major_formatter(FuncFormatter(hundred_formatter))
    ax.set_ylabel("Percentage")


def plot_cis(ax, data):
    data["ci95hi"] = data["value"] + 1.96 * (
        numpy.sqrt(data["value"] * (1 - data["value"]) / data["denominator"])
    )
    data["ci95lo"] = data["value"] - 1.96 * (
        numpy.sqrt(data["value"] * (1 - data["value"]) / data["denominator"])
    )
    ax.fill_between(
        data.index,
        (data["ci95lo"]),
        (data["ci95hi"]),
        alpha=0.1,
    )


def is_bool_as_int(series):
    """Does series have bool values but an int dtype?"""
    # numpy.nan will ensure an int series becomes a float series, so we need to
    # check for both int and float
    if not pandas.api.types.is_bool_dtype(
        series
    ) and pandas.api.types.is_numeric_dtype(series):
        series = series.dropna()
        return ((series == 0) | (series == 1)).all()
    elif not pandas.api.types.is_bool_dtype(
        series
    ) and pandas.api.types.is_object_dtype(series):
        try:
            series = series.astype(int)
        except ValueError:
            return False
        series = series.dropna()
        return ((series == 0) | (series == 1)).all()
    else:
        return False


def reorder_dataframe(measure_table, first):
    """Reorder the dataframe with rows with name matching 'first' first"""
    copy = measure_table.copy()
    copy["sorter"] = measure_table.name == first
    copy = copy.sort_values("sorter", ascending=False)
    copy = copy.drop("sorter", axis=1)
    return copy


def reorder_labels(labels):
    """
    If a list of categories has dashes, there may be numbers
    Numbers with dashes could be out of order
    Assuming a dash implies a range, sort the list by the max number
    contained in each string
    Strings with no numbers will be sorted last
    """
    max_val = {}
    if any("-" in label for label in labels):
        for label in labels:
            matches = re.findall(r"\d+", label)
            if matches:
                highest = max([int(m) for m in matches])
            else:
                highest = math.inf
            max_val[label] = highest
        return dict(sorted(max_val.items(), key=operator.itemgetter(1))).keys()
    return labels


def write_deciles_table(measure_table, output_dir, filename):
    """
    Output a deciles table, including a count of practices
    per time period to ensure enough practices per decile
    """

    num_practices = measure_table.groupby("date")["group"].count()
    num_practices = num_practices.apply(lambda x: round_values(x, 10))
    deciles_table = charts.add_percentiles(
        measure_table,
        period_column="date",
        column="value",
    )
    deciles_table = deciles_table.set_index("date")
    deciles_table["practice_count_per_date"] = num_practices
    deciles_table.to_csv(
        output_dir / f"deciles_table_{filename}.csv", index=True
    )


def add_deciles_plot(
    measure_table, ax, output_dir, filename, column_to_plot, lgd_params
):
    """
    Adds a deciles plot to the existing figure
    NOTE: NOT IDEAL as the deciles_chart command changes the overall plt
    format with commands such as plt.gcf().autofmt_xdate()
    Manually adjust the legend, label size and roation to match
    Retuns the legend so it can be included in the layout
    """
    numeric = coerce_numeric(measure_table)
    numeric = drop_zero_denominator_rows(numeric)

    write_deciles_table(numeric, output_dir, filename)

    charts.deciles_chart(
        numeric,
        period_column="date",
        column=column_to_plot,
        show_legend=True,
        ax=ax,
    )
    ax.legend(**lgd_params)


def plot_axis(
    panel_group_data,
    ax,
    column_to_plot,
    stack_years,
    date_lines,
    ci,
    lgd_params,
):
    """
    Within a figure, code to plot a single axis as a line chart
    """
    # We need to sort by date before setting it as index
    # If a 'first' group was specified, date could be out of order
    panel_group_data = panel_group_data.sort_values("date")
    panel_group_data = panel_group_data.set_index("date")
    is_bool = is_bool_as_int(panel_group_data.group)
    if is_bool:
        panel_group_data.group = panel_group_data.group.astype(int).astype(
            bool
        )
    numeric = coerce_numeric(panel_group_data)
    group_by = ["group"]
    # TODO: we may need to handle weekly data differently
    if stack_years and len(numeric.group.unique()) == 1:
        numeric = numeric.reset_index()
        numeric["year"] = numeric["date"].dt.year
        numeric["date"] = numeric["date"].dt.month_name()
        numeric = numeric.set_index("date")
        group_by = ["group", "year"]
    for plot_group, plot_group_data in numeric.groupby(group_by):
        ax.plot(
            plot_group_data.index,
            plot_group_data[column_to_plot],
            label=plot_group,
        )
        if ci:
            plot_cis(ax, plot_group_data)
    # NOTE: Use the last group_by, which will be "year" for stack_years
    labels = reorder_labels(list(numeric[group_by[-1]].astype(str).unique()))
    ax.legend(
        labels,
        **lgd_params,
    )
    if date_lines:
        min_date = min(panel_group_data.index)
        max_date = max(panel_group_data.index)
        add_date_lines(date_lines, min_date, max_date)


def get_group_chart(
    measure_table,
    first,
    column_to_plot,
    stack_years=False,
    columns=2,
    date_lines=None,
    scale=None,
    ci=None,
    exclude_group=None,
    output_dir=None,
):
    # NOTE: constrained_layout=True available in matplotlib>=3.5
    figure = plt.figure(figsize=(columns * 6, columns * 5))
    lgd_params = {
        "bbox_to_anchor": (1, 1),
        "loc": "upper left",
        "fontsize": "x-small",
        "ncol": 1,
    }

    if first:
        # NOTE: key param is in pandas>1.0
        # measure_table = measure_table.sort_values(
        #     by="name", key=lambda x: x == first, ascending=False
        # )
        measure_table = reorder_dataframe(measure_table, first)

    repeated = autoselect_labels(measure_table["name"])
    groups = measure_table.groupby("name", sort=False)
    total_plots = len(groups)

    if total_plots > 10:
        raise Exception(f"Trying to plot more than 10 plots ({total_plots})")

    # Compute Rows required
    rows = total_plots // columns
    if total_plots % columns > 0:
        rows = rows + 1

    lgds = []
    for index, panel in enumerate(groups):
        sns.set_style("darkgrid")
        # set the color palette using matplotlib
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colour_palette)

        panel_group, panel_group_data = panel
        ax = figure.add_subplot(rows, columns, index + 1)
        ax.autoscale(enable=True, axis="y")
        title = translate_group(
            panel_group_data.category.unique()[0],
            panel_group,
            repeated,
            autolabel=True,
        )
        ax.set_title(title)
        # Filter out group, but ignore case
        if exclude_group:
            panel_group_data = panel_group_data[
                panel_group_data.group.str.lower() != exclude_group.lower()
            ]

        if "practice" in panel_group:
            add_deciles_plot(
                panel_group_data,
                ax,
                output_dir,
                panel_group,
                column_to_plot,
                lgd_params,
            )

        else:
            plot_axis(
                panel_group_data,
                ax,
                column_to_plot,
                stack_years,
                date_lines,
                ci,
                lgd_params,
            )
        lgd = ax.get_legend()
        lgds.append(lgd)

        # Global plot settings
        if scale == "percentage":
            scale_hundred(ax)
        elif scale == "rate":
            scale_thousand(ax)
        if column_to_plot == "numerator":
            ax.set_ylabel("Count")
    if exclude_group:
        plt.xlabel(
            f"*Those with '{exclude_group}' category excluded from each plot"
        )
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize="small")
    ax.yaxis.label.set_alpha(1.0)
    ax.yaxis.label.set_fontsize("small")
    # Deciles chart code globally calls plt.gcf().autofmt_xdate()
    # So we have to turn the axes back on here
    for ax in plt.gcf().get_axes():
        ax.tick_params(labelbottom=True)
        ax.get_xticklabels("auto")
    plt.subplots_adjust(wspace=0.7, hspace=0.6)
    return (plt, lgds)


def get_path(*args):
    return pathlib.Path(*args).resolve()


def add_date_lines(vlines, min_date, max_date):
    for date in vlines:
        date_obj = pandas.to_datetime(date)
        if date_obj >= min_date and date_obj <= max_date:
            try:
                plt.axvline(x=date_obj, color="orange", ls="--")
            except parser._parser.ParserError:
                continue


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
    measures_group = parser.add_mutually_exclusive_group(required=False)
    measures_group.add_argument(
        "--measures-pattern",
        required=False,
        help="Glob pattern for matching one or more measures names",
    )
    measures_group.add_argument(
        "--measures-list",
        required=False,
        action="append",
        help="A list of one or more measure names",
    )
    parser.add_argument(
        "--first",
        required=False,
        help="Measures pattern for plot that should appear first",
    )
    parser.add_argument(
        "--column-to-plot",
        choices=["numerator", "value"],
        default="value",
        help="Which measure column to plot",
    )
    parser.add_argument(
        "--stack-years",
        action="store_true",
        help="If there is only one group, option to stack the years",
    ),
    parser.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Name for panel plot",
    )
    parser.add_argument(
        "--date-lines",
        nargs="+",
        help="Vertical date lines",
    )
    choices = ["percentage", "rate"]
    parser.add_argument("--scale", default=None, choices=choices)
    parser.add_argument(
        "--confidence-intervals",
        action="store_true",
        help="""
             Add confidence intervals to the plot.
             NOTE: only supported for single group",
             """,
    )
    parser.add_argument(
        "--exclude-group",
        help="Exclude group with this label from plot, e.g. Unknown",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input_file
    practice_file = args.practice_file
    measures_pattern = args.measures_pattern
    measures_list = args.measures_list
    first = args.first
    column_to_plot = args.column_to_plot
    stack_years = args.stack_years
    output_dir = args.output_dir
    output_name = args.output_name
    date_lines = args.date_lines
    scale = args.scale
    confidence_intervals = args.confidence_intervals
    exclude_group = args.exclude_group

    measure_table = get_measure_tables(input_file)
    if practice_file:
        practice_table = get_measure_tables(practice_file)
        practice_table = practice_table[practice_table.category == "practice"]
        measure_table = pandas.concat([measure_table, practice_table])

    plot_title = filename_to_title(output_name)

    # Parse the names field to determine which subset to use
    subset = subset_table(measure_table, measures_pattern, measures_list)
    chart, lgds = get_group_chart(
        subset,
        first=first,
        column_to_plot=column_to_plot,
        stack_years=stack_years,
        columns=2,
        date_lines=date_lines,
        scale=scale,
        ci=confidence_intervals,
        exclude_group=exclude_group,
        output_dir=output_dir,
    )
    write_group_chart(chart, lgds, output_dir / output_name, plot_title)
    chart.close()


if __name__ == "__main__":
    main()
