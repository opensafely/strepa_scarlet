import re
import json
import pandas as pd
import numpy as np
import fnmatch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
from IPython.display import display, Markdown, HTML, Image
import matplotlib.ticker as ticker

colour_palette = sns.color_palette("Paired", 12)
ticker.Locator.MAXTICKS = 10000

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
    "sore_throat_tonsillitis": "codelists/user-chriswood-group-a-streptococcal-sore-throat.csv",
    "invasive_strep_a": "codelists/user-chriswood-invasive-group-a-strep.csv",
}


def get_codelist_dict():
    with open(Path.cwd().parent.parent / "codelists" / "codelists.json") as f:
        data = json.load(f)
    return data["files"]


def save_to_json(d, filename: str):
    """Saves dictionary to json file"""
    with open(filename, "w") as f:
        json.dump(d, f)


def match_input_files(file: str) -> bool:
    """Checks if file name has format outputted by cohort extractor"""
    pattern = r"^input_report_([a-zA-Z]+\_)*20\d\d-(0[1-9]|1[012])-(0[1-9]|[12][0-9]|3[01])\.csv.gz"
    return True if re.match(pattern, file) else False


def get_date_input_file(file: str) -> str:
    """Gets the date in format YYYY-MM-DD from input file name string"""
    # check format
    if not match_input_files(file):
        raise Exception("Not valid input file format")

    else:
        date = re.search(r"input_report_(.*).csv.gz", file)
        return date.group(1)


def round_values(x, base=10, redact=False, redaction_threshold=5):
    """
    Rounds values to nearest multiple of base.  If redact is True, values less than or equal to
    redaction_threshold are converted to np.nan.
    Args:
        x: Value to round
        base: Base to round to
        redact: Boolean indicating if values less than redaction_threshold should be
        redacted
        redaction_threshold: Threshold for redaction
        Returns:
            Rounded value
    """

    if isinstance(x, (int, float)):
        if np.isnan(x):
            rounded = np.nan

        else:
            if redact and x <= redaction_threshold:
                rounded = np.nan

            else:
                rounded = int(base * round(x / base))
    return rounded


def plot_measures(
    df,
    filename: str,
    column_to_plot: str,
    y_label: str,
    as_bar: bool = False,
    category: str = None,
    frequency: str = "month",
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

    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(18, 8))

    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colour_palette)

    # NOTE: finite filter for dummy data
    y_max = df[np.isfinite(df[column_to_plot])][column_to_plot].max() * 1.05
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
    elif frequency == "week":
        xticks = pd.date_range(
            start=df_copy.index.min(), end=df_copy.index.max(), freq="W-THU"
        )
        ax.set_xticks(xticks)
        ax.set_xticklabels([x.strftime("%d-%m-%Y") for x in xticks])

    plt.ylabel(y_label)
    plt.xlabel("Date")
    plt.xticks(rotation="vertical")
    plt.xticks(fontsize=8)

    plt.ylim(
        bottom=0,
        top=1000 if df_copy[column_to_plot].isnull().values.all() else y_max,
    )

    if category:
        plt.legend(
            sorted(df_copy[category].unique()),
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
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


def drop_zero_denominator_rows(measure_table):
    """
    Zero-denominator rows could cause the deciles to be computed incorrectly, so should
    be dropped beforehand. For example, a practice can have zero registered patients. If
    the measure is computed from the number of registered patients by practice, then
    this practice will have a denominator of zero and, consequently, a value of inf.
    Depending on the implementation, this practice's value may be sorted as greater than
    other practices' values, which may increase the deciles.
    """
    # It's non-trivial to identify the denominator column without the associated Measure
    # instance. It's much easier to test the value column for inf, which is returned by
    # Pandas when the second argument of a division operation is zero.
    is_not_inf = measure_table["value"] != np.inf
    num_is_inf = len(is_not_inf) - is_not_inf.sum()
    return measure_table[is_not_inf].reset_index(drop=True)


def filename_to_title(filename):
    return filename.replace("_", " ").title()


def autoselect_labels(measures_list):
    measures_set = set(measures_list)
    counts = Counter(
        np.concatenate([item.split("_") for item in measures_set])
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


# NOTE: These paths will only work for notebook generation, which is run on /workspace
REPORT_DIR = Path.cwd().parent.parent / "output/report"
RESULTS_DIR = REPORT_DIR / "results"
WEEKLY_RESULTS_DIR = REPORT_DIR / "weekly/results"


def display_event_counts(file, period, dir=RESULTS_DIR):
    """
    Displays event counts table. Input is a json file containing a dictionary
    of event counts.
    """

    column_name_map = {
        "total_events": "Total recorded events",
        "total_patients": "Total unique patients with an event",
        "events_in_latest_period": f"Events in the latest {period}",
    }

    with open(f"{dir}/{file}") as f:
        event_summary = json.load(f)

        event_summary = {
            column_name_map.get(key, key): value
            for key, value in event_summary.items()
        }

        event_summary_table = pd.DataFrame(event_summary, index=[0])

    event_summary_table = event_summary_table.applymap(
        lambda x: "{:,}".format(x) if isinstance(x, int) else x
    )

    display(HTML(event_summary_table.to_html(index=False)))


def display_table(file, dir=RESULTS_DIR):
    table = pd.read_csv(f"{dir}/{file}")
    display(HTML(table.to_html(index=False)))


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
    df["Count of patients with code"] = df[
        "Count of patients with code"
    ].apply(lambda x: "{:,}".format(x))
    display(HTML(df.to_html(index=False)))


def display_standard_medicine():
    display(
        HTML(
            "<details open style='border: 1px solid #aaa;padding: 0.5em 0.5em 0.5em'><b>Counts</b>: represent patients with at least one prescription event in that week. Patients with more than one of the same prescription in a week were only counted once. Counts <=5 were redacted and all numbers were rounded to the nearest 10.<br><b>Rates</b>: divide the count by the included study population and multiply by 1,000 to achieve a rate per 1,000 registered patients.<br><b>Note</b>: Prescribing data is based on prescriptions issued within the Electronic Health Record. Prescriptions may not always be dispensed or in some cases the dispensed item may differ from the prescribed item due to the use of a <a href='https://www.nhsbsa.nhs.uk/pharmacies-gp-practices-and-appliance-contractors/serious-shortage-protocols-ssps'>Serious Shortage Protocol</a><br><b>Note</b>: Weeks run from Thursday to Wednesday to enable the extraction of the most up-to-date data</details>"
        )
    )


def display_standard_clinical():
    display(
        HTML(
            "<details open style='border: 1px solid #aaa;padding: 0.5em 0.5em 0.5em'><b>Counts</b>: represent patients with at least one clinical event in that week. Patients with more than one of the same clinical event in a week were only counted once. Counts <=5 were redacted and all numbers were rounded to the nearest 10.<br><b>Rates</b>: divide the count by the included study population and multiply by 1,000 to achieve a rate per 1,000 registered patients.<br><b>Note</b>: Clinical events data is based on a clinical code being added to a patient's record. This is often added by a clinician during a consultation to indicate the presence of a sign/symptom (e.g. sore throat) or that a clinical diagnosis has been made (e.g. Scarlet Fever). These codes do not necessarily indicate positive test results.<br><b>Note</b>: Weeks run from Thursday to Wednesday to enable the extraction of the most up-to-date data.</details>"
        )
    )


def display_medicine(
    medicine_path, medicine_name, start_date, end_date, results_dir
):
    CODELIST_DICT = get_codelist_dict()
    try:
        local_codelist = MEDICATION_TO_CODELIST[medicine_path]
        codelist_name = Path(local_codelist).parts[-1]
        codelist_url = CODELIST_DICT[codelist_name]["url"]
    except KeyError as e:
        raise e

    display(
        Markdown(f"### {medicine_name.title()} [(Codelist)]({codelist_url})")
    )
    display(
        Markdown(
            f"The below charts show patients prescribed {medicine_name} between {start_date} and {end_date}. The codelist used to identify {medicine_name} is [here]({codelist_url})."
        )
    )
    display_standard_medicine()
    display(
        Markdown(
            f"The following table shows the top 5 used codes within the {medicine_name} codelist after summing code usage over the entire study period. Codes with low usage that would have been redacted have been grouped into the category 'Other'. The proportion was computed after rounding. If more than 5 codes in the codelist are used, the proportion will not add up to 100%."
        )
    )
    display_top_5(
        f"top_5_code_table_event_code_{medicine_path}_rate.csv",
        dir=WEEKLY_RESULTS_DIR,
    )
    display(
        Markdown(
            "The second chart illustrates top code usage over time. Codes that were in the top 5 either in the first or last week of the study period were included."
        )
    )
    display_image(
        f"event_code_{medicine_path}_rate_top_5_codes_over_time.png",
        dir=WEEKLY_RESULTS_DIR,
    )
    display(
        Markdown(
            f"The below charts show the weekly count and rate of patients with recorded {medicine_name} events across the study period, with a breakdown by key demographic subgroups."
        )
    )
    display(Markdown("##### Count"))
    display_image(
        f"{medicine_path}_by_subgroup_count.png", dir=WEEKLY_RESULTS_DIR
    )
    display(Markdown("##### Rate"))
    display_image(f"{medicine_path}_by_subgroup.png", dir=WEEKLY_RESULTS_DIR)


def display_clinical(
    clinical_path,
    clinical_name,
    start_date,
    end_date,
    results_dir,
    include_minimum=False,
):
    CODELIST_DICT = get_codelist_dict()
    try:
        local_codelist = CLINICAL_TO_CODELIST[clinical_path]
        codelist_name = Path(local_codelist).parts[-1]
        codelist_url = CODELIST_DICT[codelist_name]["url"]
    except KeyError as e:
        raise e

    display(
        Markdown(f"### {clinical_name.title()} [(Codelist)]({codelist_url})")
    )
    display(
        Markdown(
            f"The below charts show patients with recorded events of {clinical_name} between {start_date} and {end_date}. The codelist used to identify {clinical_name} is [here]({codelist_url})."
        )
    )
    display_standard_clinical()
    display(
        Markdown(
            f"The following table shows the 5 most used codes within the {clinical_name} codelist after summing code usage over the entire study period. Codes with low usage that would have been redacted have been grouped into the category 'Other'. The proportion was computed after rounding. If more than 5 codes in the codelist are used, the proportion will not add up to 100%."
        )
    )
    display_top_5(
        f"top_5_code_table_event_code_{clinical_path}_rate.csv",
        dir=WEEKLY_RESULTS_DIR,
    )
    if not include_minimum:
        display(
            Markdown(
                "The second chart illustrates top code usage over time. Codes that were in the top 5 either in the first or last week of the study period were included."
            )
        )
        display_image(
            f"event_code_{clinical_path}_rate_top_5_codes_over_time.png",
            dir=WEEKLY_RESULTS_DIR,
        )
        display(
            Markdown(
                f"The below charts show the weekly count and rate of patients with recorded {clinical_name} events across the study period, with a breakdown by key demographic subgroups."
            )
        )
        display(Markdown("##### Count"))
        display_image(
            f"{clinical_path}_by_subgroup_count.png", dir=WEEKLY_RESULTS_DIR
        )
        display(Markdown("##### Rate"))
        display_image(
            f"{clinical_path}_by_subgroup.png", dir=WEEKLY_RESULTS_DIR
        )
