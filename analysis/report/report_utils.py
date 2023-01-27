import re
import json
import pandas as pd
import matplotlib.pyplot as plt
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
    plt.figure(figsize=(18, 8))
    y_max = df[column_to_plot].max() * 1.05
    # Ignore timestamp - this could be done at load time
    df["date"] = df["date"].dt.date
    df = df.set_index("date")
    if category:
        if as_bar:
            df.pivot(columns=category, values=column_to_plot).plot.bar(
                stacked=True
            )
            y_max = df.groupby(["date"])[column_to_plot].sum().max() * 1.05
        else:
            df.groupby(category)[column_to_plot].plot()
    else:
        if as_bar:
            df[column_to_plot].bar(legend=False)
        else:
            df[column_to_plot].plot(legend=False)

    plt.ylabel(y_label)
    plt.xlabel("Date")
    plt.xticks(rotation="vertical")
    plt.ylim(
        bottom=0,
        top=1000 if df[column_to_plot].isnull().values.all() else y_max,
    )

    if category:
        plt.legend(
            sorted(df[category].unique()),
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
