import argparse
import pathlib
import pandas
import numpy
import re
import operator
import math
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from report_utils import (
    parse_date,
    add_date_lines,
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
    set_fontsize,
)
import matplotlib.ticker as ticker

ticker.Locator.MAXTICKS = 10000


def scale_thousand(ax):
    """
    Scale a proportion for rate by 1000
    Used for y axis display
    """

    def thousand_formatter(x, pos):
        return f"{x*1000: .2f}"

    ax.yaxis.set_major_formatter(FuncFormatter(thousand_formatter))
    ax.set_ylabel("Rate per thousand patients")


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


def reorder_dataframe(measure_table, order):
    """Reorder the dataframe sorting first by the provided order
    This will ignore provided strings that do not match the data"""
    all_categories = set(measure_table.category.unique())
    remaining = all_categories - set(order)

    complete_order = order + list(remaining)
    order_dict = {x: index for index, x in enumerate(complete_order)}

    copy = measure_table.copy()
    copy["sorter"] = copy["category"].map(order_dict)
    copy = copy.sort_values("sorter", ascending=True)
    copy = copy.drop("sorter", axis=1)
    return copy


def reorder_labels(labels):
    """
    If a list of categories has dashes, there may be numbers
    Numbers with dashes could be out of order
    Assuming a dash implies a range, sort the list by the max number
    contained in each string
    Strings with no numbers will be sorted last
    Keep the handle/label paired together, but once reordered return two lists
    """
    max_val = {}
    if any("-" in label for handle, label in labels):
        for handle, label in labels:
            matches = re.findall(r"\d+", label)
            if matches:
                highest = max([int(m) for m in matches])
            else:
                highest = math.inf
            max_val[(handle, label)] = highest
        return zip(
            *dict(sorted(max_val.items(), key=operator.itemgetter(1))).keys()
        )
    return zip(*labels)


# NOTE: the following functions have been copied from
# https://github.com/ebmdatalab/datalab-pandas/charts.py and modified
# due to a bug where the 6th outer decile is repeated as a result of
# floating point precision in numpy.arrange
def add_percentiles(
    df, period_column=None, column=None, show_outer_percentiles=True
):
    """For each period in `period_column`, compute percentiles across that
    range.

    Adds `percentile` column.

    """
    deciles = numpy.arange(0.1, 1, 0.1)
    bottom_percentiles = numpy.arange(0.01, 0.1, 0.01)
    top_percentiles = numpy.arange(0.91, 1, 0.01)
    if show_outer_percentiles:
        quantiles = numpy.concatenate(
            (deciles, bottom_percentiles, top_percentiles)
        )
    else:
        quantiles = deciles
    df = df.groupby(period_column)[column].quantile(quantiles).reset_index()
    df = df.rename(index=str, columns={"level_1": "percentile"})
    # create integer range of percentiles
    df["percentile"] = df["percentile"].apply(lambda x: round(x * 100))
    return df


def deciles_chart(
    df,
    period_column=None,
    column=None,
    title="",
    ylabel="",
    show_outer_percentiles=True,
    show_legend=True,
    ax=None,
):
    """period_column must be dates / datetimes"""
    sns.set_style("whitegrid", {"grid.color": ".9"})
    if not ax:
        fig, ax = plt.subplots(1, 1)
    df = add_percentiles(
        df,
        period_column=period_column,
        column=column,
        show_outer_percentiles=show_outer_percentiles,
    )
    linestyles = {
        "decile": {
            "color": "b",
            "line": "b--",
            "linewidth": 1,
            "label": "decile",
        },
        "median": {
            "color": "b",
            "line": "b-",
            "linewidth": 1.5,
            "label": "median",
        },
        "percentile": {
            "color": "b",
            "line": "b:",
            "linewidth": 0.8,
            "label": "1st-9th, 91st-99th percentile",
        },
    }
    label_seen = []
    for percentile in range(1, 100):  # plot each decile line
        data = df[df["percentile"] == percentile]
        add_label = False

        if percentile == 50:
            style = linestyles["median"]
            add_label = True
        elif show_outer_percentiles and (percentile < 10 or percentile > 90):
            style = linestyles["percentile"]
            if "percentile" not in label_seen:
                label_seen.append("percentile")
                add_label = True
        else:
            style = linestyles["decile"]
            if "decile" not in label_seen:
                label_seen.append("decile")
                add_label = True
        if add_label:
            label = style["label"]
        else:
            label = "_nolegend_"

        ax.plot(
            data[period_column],
            data[column],
            style["line"],
            linewidth=style["linewidth"],
            color=style["color"],
            label=label,
        )
    ax.set_ylabel(ylabel, size=15, alpha=0.6)
    if title:
        ax.set_title(title, size=18)
    # set ymax across all subplots as largest value across dataset
    ax.set_ylim([0, df[column].max() * 1.05])
    ax.tick_params(labelsize=12)
    ax.set_xlim(
        [df[period_column].min(), df[period_column].max()]
    )  # set x axis range as full date range

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%B %Y"))
    if show_legend:
        ax.legend(
            bbox_to_anchor=(1.1, 0.8),  # arbitrary location in axes
            #  specified as (x0, y0, w, h)
            loc=6,  # which part of the bounding box should
            #  be placed at bbox_to_anchor
            ncol=1,  # number of columns in the legend
            fontsize=12,
            borderaxespad=0.0,
        )  # padding between the axes and legend
        #  specified in font-size units
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    plt.gcf().autofmt_xdate()
    return plt


def write_deciles_table(measure_table, output_dir, filename):
    """
    Output a deciles table, including a count of practices
    per time period to ensure enough practices per decile
    """

    num_practices = measure_table.groupby("date")["group"].count()
    num_practices = num_practices.apply(lambda x: round_values(x, 10))
    deciles_table = add_percentiles(
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

    deciles_chart(
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
    # If an 'order' was specified, date could be out of order
    panel_group_data = panel_group_data.sort_values("date")
    panel_group_data = panel_group_data.set_index("date")
    is_bool = is_bool_as_int(panel_group_data.group)
    if is_bool:
        panel_group_data.group = panel_group_data.group.astype(int).astype(
            bool
        )
    numeric = coerce_numeric(panel_group_data)
    group_by = "group"
    # TODO: we may need to handle weekly data differently
    if stack_years and len(numeric.group.unique()) == 1:
        numeric = numeric.reset_index()
        numeric["year"] = numeric["date"].dt.year
        numeric["date"] = numeric["date"].dt.month_name()
        numeric = numeric.set_index("date")
        group_by = ["group", "year"]
    for plot_group, plot_group_data in numeric.groupby(group_by):
        if isinstance(plot_group, tuple):
            label = plot_group[1]
        else:
            label = plot_group
        ax.plot(
            plot_group_data.index,
            plot_group_data[column_to_plot],
            label=label,
        )
        if ci:
            plot_cis(ax, plot_group_data)
    # NOTE: Use the last group_by, which will be "year" for stack_years
    handles, labels = ax.get_legend_handles_labels()
    handles_reordered, labels_reordered = reorder_labels(
        list(zip(handles, labels))
    )
    ax.legend(handles_reordered, labels_reordered, **lgd_params)
    if date_lines:
        min_date = min(panel_group_data.index)
        max_date = max(panel_group_data.index)
        add_date_lines(date_lines, min_date, max_date)


def get_group_chart(
    measure_table,
    order,
    column_to_plot,
    stack_years=False,
    columns=2,
    date_lines=None,
    scale=None,
    ci=None,
    exclude_group=None,
    output_dir=None,
    frequency="month",
    xtick_frequency=1,
):
    # NOTE: constrained_layout=True available in matplotlib>=3.5
    figure = plt.figure(figsize=(columns * 6, columns * 5))
    lgd_params = {
        "bbox_to_anchor": (1, 1),
        "loc": "upper left",
        "fontsize": "10",
        "ncol": 1,
    }

    if order:
        measure_table = reorder_dataframe(measure_table, order)

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
                panel_group_data.group.astype(str).apply(lambda x: x.lower())
                != exclude_group.lower()
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
            ax.set_ylabel("Count of patients")
    if exclude_group:
        plt.xlabel(
            f"*Those with '{exclude_group}' category excluded from each plot"
        )
    # Deciles chart code globally calls plt.gcf().autofmt_xdate()
    # So we have to turn the axes back on here
    for ax in plt.gcf().get_axes():
        ax.tick_params(labelbottom=True)
        ax.get_xticklabels("auto")
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=7, rotation=90)
        ax.tick_params(axis="y", labelsize="small")
        ax.yaxis.label.set_alpha(1.0)
        ax.yaxis.label.set_fontsize("small")

        # TODO: this will apply the same date range limits to each axis
        if not stack_years and frequency == "month":
            xticks = pandas.date_range(
                start=measure_table["date"].min(),
                end=measure_table["date"].max(),
                freq="MS",
            )
            ax.set_xticks(xticks)
            ax.set_xticklabels([x.strftime("%B %Y") for x in xticks])
            for index, label in enumerate(ax.xaxis.get_ticklabels()):
                if index % xtick_frequency != 0:
                    label.set_visible(False)
        elif not stack_years and frequency == "week":
            # show 1 tick per week
            xticks = pandas.date_range(
                start=measure_table["date"].min(),
                end=measure_table["date"].max(),
                freq="W-THU",
            )
            ax.set_xticks(xticks)
            ax.set_xticklabels([x.strftime("%d-%m-%Y") for x in xticks])

    plt.subplots_adjust(wspace=0.7, hspace=0.7)
    return (plt, lgds)


def get_path(*args):
    return pathlib.Path(*args).resolve()


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
        "--order",
        required=False,
        nargs="+",
        default=[],
        help="List of categories for subplot order",
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
        type=parse_date,
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
    parser.add_argument(
        "--xtick-frequency",
        help="Display every nth xtick",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--base-fontsize",
        help="Default text size",
        type=int,
        default=10,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input_file
    practice_file = args.practice_file
    measures_pattern = args.measures_pattern
    measures_list = args.measures_list
    order = args.order
    column_to_plot = args.column_to_plot
    stack_years = args.stack_years
    output_dir = args.output_dir
    output_name = args.output_name
    date_lines = args.date_lines
    scale = args.scale
    confidence_intervals = args.confidence_intervals
    exclude_group = args.exclude_group
    xtick_frequency = args.xtick_frequency
    base_fontsize = args.base_fontsize

    output_dir.mkdir(parents=True, exist_ok=True)
    set_fontsize(base_fontsize)

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
        order=order,
        column_to_plot=column_to_plot,
        stack_years=stack_years,
        columns=2,
        date_lines=date_lines,
        scale=scale,
        ci=confidence_intervals,
        exclude_group=exclude_group,
        output_dir=output_dir,
        xtick_frequency=xtick_frequency,
    )
    write_group_chart(chart, lgds, output_dir / output_name, plot_title)
    chart.close()


if __name__ == "__main__":
    main()
