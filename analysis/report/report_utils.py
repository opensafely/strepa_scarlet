import re
import json
import pandas as pd
import numpy
import fnmatch
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from IPython.display import display, HTML, Image


MEDICATION_TO_CODELIST = {
    "amoxicillin": "codelists/opensafely-amoxicillin-oral.csv",
    "azithromycin": "codelists/opensafely-azithromycin-oral.csv",
    "clarithromycin": "codelists/opensafely-clarithromycin-oral.csv",
    "erythromycin": "codelists/opensafely-erythromycin-oral.csv",
    "phenoxymethylpenicillin": "codelists/opensafely-phenoxymethylpenicillin-oral-preparations-only.csv",
    "cefalexin": "codelists/opensafely-cefalexin-oral.csv",
    "co_amoxiclav": "codelists/opensafely-co-amoxiclav-oral.csv",
    "flucloxacillin": "codelists/opensafely-flucloxacillin-oral.csv",
}

CLINICAL_TO_CODELIST = {
    "scarlet_fever": "codelists/user-chriswood-scarlet-fever.csv",
    "strep_a_sore_throat": "codelists/user-chriswood-group-a-streptococcal-sore-throat.csv",
    "invasive_strep_a": "codelists/user-chriswood-invasive-group-a-strep.csv",
    "sore_throat_prediction": "codelists/user-chriswood-sore-throat-clinical-prediction-rules.csv",
    "throat_swab": "codelists/user-chriswood-throat-swab.csv",
}


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
    category within the category column. Saves output as jpeg file.
    Args:
        df: A measure table
        column_to_plot: Column name for y-axis values
        y_label: Label to use for y-axis
        as_bar: Boolean indicating if bar chart should be plotted instead of line chart. Only valid if no categories.
        category: Name of column indicating different categories
    """
    df_copy = df.copy()
    plt.figure(figsize=(18, 8))
    y_max = df[column_to_plot].max() * 1.05
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
            ax = df_copy.pivot(
                columns=category, values=column_to_plot
            ).plot.bar(stacked=True)
            y_max = (
                df_copy.groupby(["date"])[column_to_plot].sum().max() * 1.05
            )
        else:
            ax = df_copy.groupby(category)[column_to_plot].plot()
    else:
        if as_bar:
            df_copy[column_to_plot].bar(legend=False)
        else:
            df_copy[column_to_plot].plot(legend=False)

    #if as_bar:
        # Matplotlib treats bar labels as necessary categories
        # So we force it to use only every third label
        #labels = ax.get_xticklabels()
        #skipped_labels = [
        #    x if index % 3 == 0 else "" for index, x in enumerate(labels)
        #]
        #ax.set_xticklabels(skipped_labels)

    plt.ylabel(y_label)
    plt.xlabel("Date")
    plt.xticks(rotation="vertical")
    plt.ylim(
        bottom=0,
        top=1000 if df_copy[column_to_plot].isnull().values.all() else y_max,
    )

    if category:
        plt.legend(
            sorted(df_copy[category].unique()),
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
            prop={"size": 6},
        )

    plt.tight_layout()

    plt.savefig(f"{filename}.jpeg")
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
    coerced["numerator"] = pd.to_numeric(coerced["numerator"], errors="coerce")
    coerced["denominator"] = pd.to_numeric(
        coerced["denominator"], errors="coerce"
    )
    coerced["value"] = pd.to_numeric(coerced["value"], errors="coerce")
    coerced["group"] = coerced["group"].astype(str)
    return coerced


def filename_to_title(filename):
    return filename.replace("_", " ").title()


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


def get_measure_tables(input_file):
    # The `date` column is assigned by the measures framework.
    measure_table = pd.read_csv(input_file, parse_dates=["date"])

    return measure_table


def match_paths(files, pattern):
    return fnmatch.filter(files, pattern)


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


def write_group_chart(group_chart, lgds, path, plot_title):
    suptitle = plt.suptitle(plot_title)
    group_chart.savefig(
        path, bbox_extra_artists=tuple(lgds) + (suptitle,), bbox_inches="tight"
    )


REPORT_DIR = Path.cwd().parent.parent / "output/report"
RESULTS_DIR = REPORT_DIR / "results"
WEEKLY_RESULTS_DIR = REPORT_DIR / "weekly/results"


def display_event_counts(file, dir=RESULTS_DIR):
    """
    Displays event counts table. Input is a json file containing a dictionary
    of event counts.
    """
    with open(f"{dir}/{file}") as f:
        event_summary = json.load(f)
        event_summary_table = pd.DataFrame(event_summary, index=[0])

    display(HTML(event_summary_table.to_html()))


def display_image(file, dir=RESULTS_DIR):
    """
    Displays image in file
    """
    display(Image(filename=f"{dir}/{file}"))


def display_top_5(file, dir=RESULTS_DIR):
    """
    Displays a pandas dataframe in a table. Input is a csv file.
    """
    df = pd.read_csv(f"{dir}/{file}")
    display(HTML(df.to_html()))
