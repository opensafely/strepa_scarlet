import operator
import re
import json
import math
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import fnmatch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
from IPython.display import display, Markdown, HTML, Image

colour_palette = sns.color_palette("Paired", 12)

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
    "scarlet_fever": "codelists/opensafely-scarlet-fever.csv",
    "sore_throat_tonsillitis": "codelists/opensafely-group-a-streptococcal-sore-throat.csv",
    "invasive_strep_a": "codelists/opensafely-invasive-group-a-strep.csv",
}

# NOTE: use dashes instead of underscores do autolabel doesn't remove
GROUPED_MEDICATIONS = {
    "group-1": [
        "phenoxymethylpenicillin",
    ],
    "group-2": [
        "flucloxacillin",
        "amoxicillin",
        "clarithromycin",
        "erythromycin",
        "azithromycin",
    ],
    "group-3": ["cefalexin", "co_amoxiclav"],
}


def set_fontsize(base_fontsize):
    plt.rc("font", size=base_fontsize)  # controls default text sizes


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


def ci_95_proportion(df, scale=1):
    # NOTE: do not assume df has value
    # See formula:
    # https://sphweb.bumc.bu.edu/otlt/MPH-Modules/PH717-QuantCore/PH717-Module6-RandomError/PH717-Module6-RandomError12.html
    cis = pd.DataFrame()
    val = df.numerator / df.denominator
    sd = np.sqrt(((val * (1 - val)) / df.denominator))
    cis[0] = scale * (val)
    cis[1] = scale * (val - 1.96 * sd)
    cis[2] = scale * (val + 1.96 * sd)
    return cis


def ci_to_str(ci_df, decimals=2):
    return ci_df.apply(
        lambda x: f"{x[0]:.{decimals}f} ({x[1]:.{decimals}f} to {x[2]:.{decimals}f})",
        axis=1,
    )


def parse_date(date_str):
    return pd.to_datetime(date_str).date()


def add_date_lines(vlines, min_date, max_date):
    for date in vlines:
        if date >= min_date and date <= max_date:
            plt.axvline(x=date, color="orange", ls="--")


def is_bool_as_int(series):
    """Does series have bool values but an int dtype?"""
    # numpy.nan will ensure an int series becomes a float series, so we need to
    # check for both int and float
    if not pd.api.types.is_bool_dtype(
        series
    ) and pd.api.types.is_numeric_dtype(series):
        series = series.dropna()
        return ((series == 0) | (series == 1)).all()
    elif not pd.api.types.is_bool_dtype(
        series
    ) and pd.api.types.is_object_dtype(series):
        try:
            series = series.astype(int)
        except ValueError:
            return False
        series = series.dropna()
        return ((series == 0) | (series == 1)).all()
    else:
        return False


def reorder_dashes(data, level=1):
    """
    Some strings (i.e. numbers that contain dashes) should be sorted
    numerically rather than alphabetically
    Assuming a dash implies a range, sort the categories by the max number
    contained in each string
    Strings with no numbers will be sorted last
    """
    max_val = {}
    level_values = data.index.get_level_values(level)
    if any("-" in label for label in level_values):
        for label in level_values:
            matches = re.findall(r"\d+", label)
            if matches:
                highest = max([int(m) for m in matches])
            else:
                highest = math.inf
            max_val[label] = highest
        d = dict(
            zip(
                dict(
                    sorted(max_val.items(), key=operator.itemgetter(1))
                ).keys(),
                range(len(data.index)),
            )
        )
        data["sorter"] = level_values.map(d)
        data = data.sort_values("sorter")
        data = data.drop("sorter", axis=1)
    return data


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
    measure_table = pd.read_csv(
        input_file,
        parse_dates=["date"],
        dtype={"numerator": float, "denominator": float, "value": float},
        na_values="[REDACTED]",
    )

    return measure_table


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


def format_season_table(df, column_to_plot, category):
    cis = ci_95_proportion(df, scale=1000)
    df["Rate (95% CI)"] = ci_to_str(cis)
    df = df.rename({"numerator": "Count"}, axis=1)

    # Reformat the year to the season
    df["gas_year_int"] = df["gas_year"].apply(lambda x: x.year)
    df["gas_year"] = df.apply(
        lambda x: f"{x.gas_year_int-1}/{x.gas_year_int % 100}", axis=1
    )
    df = df.drop("gas_year_int", axis=1)

    table = df.pipe(
        MultiIndex_pivot,
        index=[category, "type"],
        columns="gas_year",
        values=["Count", "value", "Rate (95% CI)"],
    )
    table.Count = table.Count.astype(float)
    table.value = table.value.astype(float)
    ratio = table["value"]["2022/23"] / table["value"]["2017/18"]

    # See formula of ci irr:
    # https://researchonline.lshtm.ac.uk/id/eprint/251164/1/pmed.1001270.s005.pdf
    sd_log_irr = np.sqrt(
        (1 / table["Count"]["2022/23"]) + (1 / table["Count"]["2017/18"])
    )
    lci = np.exp(np.log(ratio) - 1.96 * sd_log_irr)
    uci = np.exp(np.log(ratio) + 1.96 * sd_log_irr)
    rr_cis = pd.concat([ratio, lci, uci], axis=1)

    rr_cis_str = ci_to_str(rr_cis)
    rr_cis_str.name = ("2023 v 2018", "Rate Ratio (95% CI)")

    # Create table with (comma delimited) count, rate (95% CI)
    table["Count"] = table["Count"].applymap(lambda x: f"{x:,.0f}")
    count_cis = (table[["Count", "Rate (95% CI)"]]).swaplevel(axis=1)
    cols = count_cis.columns.get_level_values(0).unique()
    reordered = count_cis.reindex(columns=cols, level="gas_year")
    return pd.concat([reordered, rr_cis_str], axis=1)


# NOTE: requires that the date is the index
def make_season_table(df, category, column_to_plot, output_dir, filename=None):
    df["gas_year"] = pd.to_datetime(df.index).to_period("A-Aug")
    mins_and_maxes = []
    # In order for sorting to work, must be numeric
    assert is_numeric_dtype(df[column_to_plot])
    for year, data in df.groupby(["gas_year", category]):
        data_sorted = data.dropna(subset=[column_to_plot]).sort_values(
            column_to_plot
        )
        if data_sorted.empty:
            continue
        data_min = data_sorted.iloc[0]
        data_max = data_sorted.iloc[-1]
        data_min["type"] = "min"
        data_max["type"] = "max"
        mins_and_maxes.append(data_min)
        mins_and_maxes.append(data_max)
    try:
        table = pd.concat(mins_and_maxes, axis=1).T
        # Concatting and transposing seems to lose types
        table.numerator = pd.to_numeric(table.numerator, errors="coerce")
        table.denominator = pd.to_numeric(table.denominator, errors="coerce")
    except ValueError:
        table = pd.DataFrame()
        formatted = pd.DataFrame()
    # If too much is redacted, we cannot make the table
    # But project yaml might expect existance of html
    if filename and not table.empty:
        try:
            formatted = format_season_table(table, column_to_plot, category)
            formatted = reorder_dashes(formatted, level=0)
        except:
            formatted = pd.DataFrame()
    if filename:
        formatted.to_html(output_dir / f"{filename}_table.html")
    return table


def annotate_seasons(season_table, column_to_plot, ax):
    # TODO: check whether this works for weekly
    # NOTE: experimenting with just displaying a marker for max
    """
    min_table = season_table[season_table.type == "min"]
    for _, data_min in min_table.iterrows():
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
        )
    """
    max_table = season_table[season_table.type == "max"]
    ax.scatter(
        max_table.index,
        max_table[column_to_plot].values,
        marker="x",
        color="k",
    )
    """
    for _, data_max in max_table.iterrows():
        plt.annotate(
            int(data_max[column_to_plot]),
            xy=(data_max.name, data_max[column_to_plot]),
            xytext=(0, -25),
            textcoords="offset points",
            verticalalignment="top",
            arrowprops=dict(facecolor="blue", shrink=0.025),
        )
    """


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
        f"{path}.jpg", bbox_extra_artists=(tuple(lgds) + (suptitle,)), bbox_inches="tight", dpi=300
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
    # Make the table fit without scrolling for export to pdf
    # Will not necessarily work if more columns are added
    style = "<style>.dataframe tr { font-size: 8pt; }</style>"
    display(HTML(style + table.to_html(index=False)))


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


def display_standard_medicine(time_period="month"):
    display_str = f"<details open style='border: 1px solid #aaa;padding: 0.5em 0.5em 0.5em'><b>Counts</b>: represent patients with at least one prescription event in that {time_period}. Patients with more than one of the same prescription in a {time_period} were only counted once. Counts <=5 were redacted and all numbers were rounded to the nearest 10.<br><b>Rates</b>: divide the count by the included study population and multiply by 1,000 to achieve a rate per 1,000 registered patients.<br><b>Note</b>: Prescribing data is based on prescriptions issued within the Electronic Health Record. Prescriptions may not always be dispensed or in some cases the dispensed item may differ from the prescribed item due to the use of a <a href='https://www.nhsbsa.nhs.uk/pharmacies-gp-practices-and-appliance-contractors/serious-shortage-protocols-ssps'>Serious Shortage Protocol</a>"
    if time_period == "week":
        display_str += "<br><b>Note</b>: Weeks run from Thursday to Wednesday to enable the extraction of the most up-to-date data"
    display_str += "</details>"
    display(HTML(display_str))


def display_standard_clinical(time_period="month"):
    display_str = f"<details open style='border: 1px solid #aaa;padding: 0.5em 0.5em 0.5em'><b>Counts</b>: represent patients with at least one clinical event in that {time_period}. Patients with more than one of the same clinical event in a {time_period} were only counted once. Counts <=5 were redacted and all numbers were rounded to the nearest 10.<br><b>Rates</b>: divide the count by the included study population and multiply by 1,000 to achieve a rate per 1,000 registered patients.<br><b>Note</b>: Clinical events data is based on a clinical code being added to a patient's record. This is often added by a clinician during a consultation to indicate the presence of a sign/symptom (e.g. sore throat) or that a clinical diagnosis has been made (e.g. Scarlet Fever). These codes do not necessarily indicate positive test results."
    if time_period == "week":
        display_str += "<br><b>Note</b>: Weeks run from Thursday to Wednesday to enable the extraction of the most up-to-date data"
    display_str += "</details>"
    display(HTML(display_str))


def display_medicine(
    medicine_path, medicine_name, start_date, end_date, time_period="month"
):
    if time_period == "week":
        results_dir = WEEKLY_RESULTS_DIR
    else:
        results_dir = RESULTS_DIR
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
    display_standard_medicine(time_period=time_period)
    display(
        Markdown(
            f"The following table shows the top 5 used codes within the {medicine_name} codelist after summing code usage over the entire study period. Codes with low usage that would have been redacted have been grouped into the category 'Other'. The proportion was computed after rounding. If more than 5 codes in the codelist are used, the proportion will not add up to 100%."
        )
    )
    display_top_5(
        f"top_5_code_table_event_code_{medicine_path}_rate.csv",
        dir=results_dir,
    )
    display(
        Markdown(
            f"The second chart illustrates top code usage over time. Codes that were in the top 5 either in the first or last {time_period} of the study period were included."
        )
    )
    display_image(
        f"event_code_{medicine_path}_rate_top_5_codes_over_time.jpg",
        dir=results_dir,
    )
    display(
        Markdown(
            f"The below charts show the {time_period}ly count and rate of patients with recorded {medicine_name} events across the study period, with a breakdown by key demographic subgroups."
        )
    )
    display(Markdown("##### Count"))
    display_image(f"{medicine_path}_by_subgroup_count.jpg", dir=results_dir)
    display(Markdown("##### Rate"))
    display_image(f"{medicine_path}_by_subgroup.jpg", dir=results_dir)
    if time_period == "month":
        display(
            Markdown(
                "##### Rate with a group A strep clinical event of interest"
            )
        )
        display(
            Markdown(
                f"The below chart shows the monthly rate, broken down by key demographic subgroups, of patients with a recorded {medicine_name} prescription event AND a record of any of the potential group A strep clinical events of interest (scarlet fever, sore throat/tonsillitis or invasive group A strep) up to 14 days prior to or 7 days after the prescription event."
            )
        )
        display_image(
            f"{medicine_path}_with_clinical_any_by_subgroup.jpg",
            dir=results_dir,
        )


def display_clinical(
    clinical_path,
    clinical_name,
    start_date,
    end_date,
    time_period="month",
    include_minimum=False,
):
    if time_period == "week":
        results_dir = WEEKLY_RESULTS_DIR
    else:
        results_dir = RESULTS_DIR
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
    display_standard_clinical(time_period=time_period)
    display(
        Markdown(
            f"The following table shows the 5 most used codes within the {clinical_name} codelist after summing code usage over the entire study period. Codes with low usage that would have been redacted have been grouped into the category 'Other'. The proportion was computed after rounding. If more than 5 codes in the codelist are used, the proportion will not add up to 100%."
        )
    )
    display_top_5(
        f"top_5_code_table_event_code_{clinical_path}_rate.csv",
        dir=results_dir,
    )
    if not include_minimum:
        display(
            Markdown(
                f"The second chart illustrates top code usage over time. Codes that were in the top 5 either in the first or last {time_period} of the study period were included."
            )
        )
        display_image(
            f"event_code_{clinical_path}_rate_top_5_codes_over_time.jpg",
            dir=results_dir,
        )
        display(
            Markdown(
                f"The below charts show the {time_period}ly count and rate of patients with recorded {clinical_name} events across the study period, with a breakdown by key demographic subgroups."
            )
        )
        display(Markdown("##### Count"))
        display_image(
            f"{clinical_path}_by_subgroup_count.jpg", dir=results_dir
        )
        display(Markdown("##### Rate"))
        display_image(f"{clinical_path}_by_subgroup.jpg", dir=results_dir)
        if time_period == "month":
            display(Markdown("##### Rate with an antibiotic of interest"))
            display(
                Markdown(
                    f"The below chart shows the monthly rate, broken down by key demographic subgroups, of patients with recorded {clinical_name} events AND a prescription for any antibiotic <a href='#prescribing'>listed in this report</a> up to 7 days prior to or 14 days after the clinical event."
                )
            )
            display_image(
                f"{clinical_path}_with_medication_any_by_subgroup.jpg",
                dir=results_dir,
            )
