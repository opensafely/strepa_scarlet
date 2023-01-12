import re
import json
import pandas
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def save_to_json(d, filename: str):
    """Saves dictionary to json file"""
    with open(filename, "w") as f:
        json.dump(d, f)

def match_input_files(file: str) -> bool:
    """Checks if file name has format outputted by cohort extractor"""
    pattern = r"^input_report_20\d\d-(0[1-9]|1[012])-(0[1-9]|[12][0-9]|3[01])\.csv.gz"
    return True if re.match(pattern, file) else False


def get_date_input_file(file: str) -> str:
    """Gets the date in format YYYY-MM-DD from input file name string"""
    # check format
    if not match_input_files(file):
        raise Exception("Not valid input file format")

    else:
        date = re.search(r"input_report_(.*).csv.gz", file)
        return date.group(1)

def plot_measures(
    df,
    filename: str,
    column_to_plot: str,
    y_label: str,
    as_bar: bool = False,
    category: str = None,
):
    """Produce time series plot from measures table.  One line is plotted for each sub
    category within the category column. Saves output in 'output' dir as jpeg file.
    Args:
        df: A measure table
        column_to_plot: Column name for y-axis values
        y_label: Label to use for y-axis
        as_bar: Boolean indicating if bar chart should be plotted instead of line chart. Only valid if no categories.
        category: Name of column indicating different categories
    """
    plt.figure(figsize=(15, 8))
    if category:
        df[category] = df[category].fillna("Missing")
        for unique_category in sorted(df[category].unique()):

            # subset on category column and sort by date
            df_subset = df[df[category] == unique_category].sort_values("date")

            plt.plot(df_subset["date"], df_subset[column_to_plot])
    else:
        if as_bar:
            df.plot.bar("date", column_to_plot, legend=False)
        else:
            plt.plot(df["date"], df[column_to_plot])

    x_labels = sorted(df["date"].unique())
    plt.ylabel(y_label)
    plt.xlabel("Date")
    plt.xticks(x_labels, rotation="vertical")
    plt.ylim(
        bottom=0,
        top=1000
        if df[column_to_plot].isnull().values.all()
        else df[column_to_plot].max() * 1.05
    )

    if category:
        plt.legend(
            sorted(df[category].unique()), bbox_to_anchor=(1.04, 1), loc="upper left"
        )

    plt.tight_layout()

    plt.savefig(f"output/{filename}.jpeg")
    plt.close()


def coerce_numeric(table):
    """
    The denominator and value columns should contain only numeric values
    Other values, such as the REDACTED string, or values introduced by error,
    should not be plotted
    Use a copy to avoid SettingWithCopyWarning
    Leave NaN values in df so missing data are not inferred
    """
    coerced = table.copy()
    coerced["numerator"] = pandas.to_numeric(
        coerced["numerator"], errors="coerce"
    )
    coerced["denominator"] = pandas.to_numeric(
        coerced["denominator"], errors="coerce"
    )
    coerced["value"] = pandas.to_numeric(coerced["value"], errors="coerce")
    return coerced
