import argparse
import pathlib
import fnmatch
import pandas
import numpy

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from dateutil import parser
from collections import Counter
from report_utils import coerce_numeric


def get_measure_tables(input_file):
    # The `date` column is assigned by the measures framework.
    measure_table = pandas.read_csv(input_file, parse_dates=["date"])

    return measure_table


def subset_table(measure_table, measures_pattern, measures_list):
    """
    Given either a pattern of list of names, extract the subset of a joined
    measures file based on the 'name' column
    """
    if measures_pattern:
        measures_list = match_paths(measure_table["name"], measures_pattern)
        if len(measures_list) == 0:
            raise ValueError("Pattern did not match any files")

    if not measures_list:
        return measure_table
    return measure_table[measure_table["name"].isin(measures_list)]


def scale_thousand(ax):
    """
    Scale a proportion for rate by 1000
    Used for y axis display
    """

    def thousand_formatter(x, pos):
        return f"{x*1000: .0f}"

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


def autoselect_labels(measures_list):
    measures_set = set(measures_list)
    counts = Counter(
        numpy.concatenate([item.split("_") for item in measures_set])
    )
    remove = [k for k, v in counts.items() if v == len(measures_set)]
    return remove


def translate_group(group_var, label, repeated, autolabel=False):
    """
    Translate a measure name into a plot label
    Autolabel uses the 'name' column, but removes words that appear in every
    item in the group. If there are no unique strings, it falls back to group
    The alternative labeling uses the 'group' column
    """
    title = " ".join(
        [x for x in label.split("_") if x not in repeated]
    ).title()
    if not title or not autolabel:
        return filename_to_title(group_var)
    else:
        return title


def filename_to_title(filename):
    return filename.replace("_", " ").title()


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


def get_group_chart(
    measure_table,
    first=None,
    columns=2,
    date_lines=None,
    scale=None,
    ci=False,
    exclude_group=None,
):
    # NOTE: constrained_layout=True available in matplotlib>=3.5
    figure = plt.figure(figsize=(columns * 6, columns * 5))
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
        panel_group, panel_group_data = panel
        # We need to sort by date before setting it as index
        # If a 'first' group was specified, date could be out of order
        panel_group_data = panel_group_data.sort_values("date")
        panel_group_data = panel_group_data.set_index("date")
        is_bool = is_bool_as_int(panel_group_data.group)
        if is_bool:
            panel_group_data.group = panel_group_data.group.astype(int).astype(
                bool
            )
        ax = figure.add_subplot(rows, columns, index + 1)
        ax.autoscale(enable=True, axis="y")
        title = translate_group(
            panel_group_data.category[0],
            panel_group,
            repeated,
            autolabel=True,
        )
        ax.set_title(title)
        filtered = panel_group_data[panel_group_data.group != exclude_group]
        numeric = coerce_numeric(filtered)
        for plot_group, plot_group_data in numeric.groupby("group"):
            ax.plot(
                plot_group_data.index, plot_group_data.value, label=plot_group
            )
            if ci:
                plot_cis(ax, plot_group_data)
            lgd = ax.legend(
                bbox_to_anchor=(1, 1),
                loc="upper left",
                fontsize="x-small",
                ncol=1,
            )
            lgds.append(lgd)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=7)
        if date_lines:
            min_date = min(measure_table.index)
            max_date = max(measure_table.index)
            add_date_lines(plt, date_lines, min_date, max_date)
        if scale == "percentage":
            scale_hundred(ax)
        elif scale == "rate":
            scale_thousand(ax)
    plt.subplots_adjust(wspace=0.5, hspace=0.4)
    return (plt, lgds)


def write_group_chart(group_chart, lgds, path, plot_title):
    suptitle = plt.suptitle(plot_title)
    group_chart.savefig(
        path, bbox_extra_artists=tuple(lgds) + (suptitle,), bbox_inches="tight"
    )


def get_path(*args):
    return pathlib.Path(*args).resolve()


def match_paths(files, pattern):
    return fnmatch.filter(files, pattern)


def add_date_lines(plt, vlines, min_date, max_date):
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
    measures_pattern = args.measures_pattern
    measures_list = args.measures_list
    first = args.first
    output_dir = args.output_dir
    output_name = args.output_name
    date_lines = args.date_lines
    scale = args.scale
    confidence_intervals = args.confidence_intervals
    exclude_group = args.exclude_group

    measure_table = get_measure_tables(input_file)

    plot_title = filename_to_title(output_name)

    # Parse the names field to determine which subset to use
    subset = subset_table(measure_table, measures_pattern, measures_list)
    chart, lgds = get_group_chart(
        subset,
        first=first,
        columns=2,
        date_lines=date_lines,
        scale=scale,
        ci=confidence_intervals,
        exclude_group=exclude_group,
    )
    write_group_chart(chart, lgds, output_dir / output_name, plot_title)
    chart.close()


if __name__ == "__main__":
    main()
