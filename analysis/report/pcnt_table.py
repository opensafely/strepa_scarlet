import pathlib
import argparse
import pandas
import fnmatch
import re
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from report_utils import ci_95_proportion, ci_to_str


def get_measure_tables(input_file):
    measure_table = pandas.read_csv(
        input_file,
        dtype={"numerator": float, "denominator": float, "value": float},
        na_values="[REDACTED]",
    )

    return measure_table


def subset_table(measure_table, measures_pattern, date):
    """
    Given either a pattern of list of names, extract the subset of a joined
    measures file based on the 'name' column
    """

    measure_table = measure_table[measure_table["date"] == date]

    measures_list = []
    for pattern in measures_pattern:
        paths_to_add = match_paths(measure_table["name"], pattern)
        if len(paths_to_add) == 0:
            raise ValueError(f"Pattern did not match any rows: {pattern}")
        measures_list += paths_to_add

    table_subset = measure_table[measure_table["name"].isin(measures_list)]

    if table_subset.empty:
        raise ValueError("Patterns did not match any rows")

    return table_subset

def plot_pcnt_over_time(measure_table, output_dir, type):
    """
    Plot the percentage over time for each measure.
    """
    measure_table['date'] = pandas.to_datetime(measure_table['date'])
    measure_table['value'] *= 100
    

    valid_dates = measure_table.dropna(subset=['value'])['date']
    start_date = valid_dates.min()
    end_date = valid_dates.max()
    measure_table = measure_table[(measure_table['date'] >= start_date) & (measure_table['date'] <= end_date)]
    
    plt.figure(figsize=(10, 6))
    for name, group in measure_table.groupby('name'):

        legend_label = name

        if type == "clinical":
            match = re.search(r"event_(\w+)_with_clinical_any_pcnt", name)
            if match:
                legend_label = match.group(1).title()
        elif type == "medication":
            match = re.search(r"event_(\w+)_with_medication_any_pcnt", name)
            if match:
                legend_label = match.group(1).title()

        plt.plot(group['date'], group['value'], label=legend_label)
    
    plt.xlabel('Date')
    plt.ylabel('Percentage')
    plt.title(f'Percentage with {type} event')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.xlim(start_date, end_date)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.tight_layout()
    plt.savefig(output_dir / f"pcnt_over_time_{type}.jpg", dpi=300)
    plt.close()

def match_paths(files, pattern):
    return fnmatch.filter(files, pattern)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to single joined measures file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="Path to the output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input_file
    output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    measure_table = get_measure_tables(input_file)

    end_date = measure_table.date.max()
    medications = subset_table(
        measure_table, ["event_*_with_clinical_any_pcnt"], end_date
    )
    medications.index = pandas.MultiIndex.from_product(
        [
            ["Medication with clinical any"],
            list(
                medications.name.str.extract(
                    r"event_(\w+)_with_clinical_any_pcnt", expand=False
                ).str.title()
            ),
        ]
    )
    medications = medications.sort_values("value", ascending=False)
    medications_cis = ci_to_str(
        ci_95_proportion(medications, scale=100), decimals=1
    )
    clinical = subset_table(
        measure_table, ["event_*_with_medication_any_pcnt"], end_date
    )
    clinical.index = pandas.MultiIndex.from_product(
        [
            ["Clinical with medication any"],
            list(
                clinical.name.str.extract(
                    r"event_(\w+)_with_medication_any_pcnt", expand=False
                ).str.title()
            ),
        ]
    )
    clinical = clinical.sort_values("value", ascending=False)
    clinical_cis = ci_to_str(ci_95_proportion(clinical, scale=100), decimals=1)
    joined = pandas.concat([medications_cis, clinical_cis])
    joined.name = "Percent (95% CI)"
    table = pandas.DataFrame(joined)
    table.to_html(output_dir / "pcnt_with_indication.html", index=True)

    filtered_table_med = measure_table[
        measure_table['name'].str.contains('with_medication_any_pcnt')
    ]   
    plot_pcnt_over_time(filtered_table_med, output_dir, "medication")

    filtered_table_clinical = measure_table[
        measure_table['name'].str.contains('with_clinical_any_pcnt')
    ]    
    plot_pcnt_over_time(filtered_table_clinical, output_dir, "clinical")
    
 

if __name__ == "__main__":
    main()
