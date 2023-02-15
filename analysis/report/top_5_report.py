import pandas as pd
import numpy as np
import argparse
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns

from report_utils import (
    coerce_numeric,
    round_values,
    MEDICATION_TO_CODELIST,
    CLINICAL_TO_CODELIST,
    colour_palette,
)


def get_measure_tables(input_file):
    measure_table = pd.read_csv(input_file, dtype=str)

    return measure_table


def write_csv(df, path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **kwargs)


def group_low_values(df, count_column, code_column, threshold):
    """Suppresses low values and groups suppressed values into
    a new row "Other".

    Args:
        df: A measure table of counts by code.
        count_column: The name of the count column in the measure table.
        code_column: The name of the code column in the codelist table.
        threshold: Redaction threshold to use
    Returns:
        A table with redacted counts
    """

    # get sum of any values <= threshold
    suppressed_count = df.loc[
        df[count_column] <= threshold, count_column
    ].sum()
    suppressed_df = df.loc[df[count_column] > threshold, count_column]

    # if suppressed values >0 ensure total suppressed count > threshold.
    # Also suppress if all values 0
    if (suppressed_count > 0) | (
        (suppressed_count == 0) & (len(suppressed_df) != len(df))
    ):

        # redact counts <= threshold
        df.loc[df[count_column] <= threshold, count_column] = np.nan

        # If all values 0, suppress them
        if suppressed_count == 0:
            df.loc[df[count_column] == 0, :] = np.nan

        else:
            # if suppressed count <= threshold redact further values
            while suppressed_count <= threshold:
                suppressed_count += df[count_column].min()
                df.loc[df[count_column].idxmin(), :] = np.nan

        # drop all rows where count column is null
        df = df.loc[df[count_column].notnull(), :]

        # add suppressed count as "Other" row (if > threshold)
        if suppressed_count > threshold:
            suppressed_count = {
                code_column: "Other",
                count_column: suppressed_count,
            }
            df = pd.concat(
                [df, pd.DataFrame([suppressed_count])], ignore_index=True
            )

    return df


def calculate_top_5(
    events_table, code_df, code_column, term_column, df_code, date_subset=None
):
    """
    Calculates the top 5 codes recorded in the measure table.
    Args:
        events_table: A measure table.
        code_df: A codelist table.
        code_column: The name of the code column in the codelist table.
        term_column: The name of the term column in the codelist table.
        measure: The measure ID.
        date_subset: The date to calculate the top 5 codes. If None, the top 5 codes are calculated across the whole period.
    """

    if date_subset:
        events_table = events_table.loc[
            (events_table["date"] == date_subset), :
        ]

    # sum event counts over dates
    events_table = (
        events_table.groupby(df_code).sum(numeric_only=True).reset_index()
    )

    # calculate % makeup of each code
    total_events = events_table["num"].sum()
    events_table["Proportion of codes (%)"] = round(
        (events_table["num"] / total_events) * 100, 2
    )

    # Gets the human-friendly description of the code for the given row
    # e.g. "Systolic blood pressure".
    code_df[code_column] = code_df[code_column].astype(str)
    code_df = code_df.set_index(code_column).rename(
        columns={term_column: "Description"}
    )

    events_table = events_table.set_index(df_code).join(code_df).reset_index()

    # set description of "Other column" to something readable
    events_table.loc[events_table[df_code] == "Other", "Description"] = "-"

    # Rename the code column to something consistent
    events_table.rename(
        columns={df_code: "Code", "num": "Count"}, inplace=True
    )

    events_table = events_table.loc[
        :, ["Code", "Description", "Count", "Proportion of codes (%)"]
    ]

    # sort by count
    events_table = events_table.sort_values("Count", ascending=False)

    return events_table


def create_top_5_code_table(
    df,
    code_df,
    code_column,
    term_column,
    low_count_threshold,
    rounding_base,
    nrows=5,
):
    """Creates tables of the top codes recorded with the number of events and %
       makeup of each code. This is calculated in the first and last date periods
       as well as across the total period.
    Args:
        df: A measure table.
        code_df: A codelist table.
        code_column: The name of the code column in the codelist table.
        term_column: The name of the term column in the codelist table.
        measure: The measure ID.
        low_count_threshold: Value to use as threshold for disclosure control.
        rounding_base: Base to round to.
        nrows: The number of rows to display.
    Returns:
        top_5_first_date_period: A table of the top codes recorded in the first date period.
        top_5_last_date_period: A table of the top codes recorded in the last date period.
        top_5_total_period: A table of the top codes recorded across the total period.
    """

    # cast both code columns to str
    df_code = "code"

    # remove rows where code is "Missing"
    df = df.loc[df[df_code] != "Missing", :]

    # sum event counts over patients
    first_date_period = df["date"].min()
    last_date_period = df["date"].max()
    event_counts = group_low_values(df, "num", df_code, low_count_threshold)
    # round
    event_counts["num"] = event_counts["num"].apply(
        lambda x: round_values(x, rounding_base)
    )

    top_5_first_date_period = calculate_top_5(
        event_counts,
        code_df,
        code_column,
        term_column,
        df_code,
        first_date_period,
    ).head(nrows)

    top_5_last_date_period = calculate_top_5(
        event_counts,
        code_df,
        code_column,
        term_column,
        df_code,
        last_date_period,
    ).head(nrows)

    top_5_whole_period = calculate_top_5(
        event_counts, code_df, code_column, term_column, df_code
    ).head(nrows)

    # return top n rows
    return top_5_first_date_period, top_5_last_date_period, top_5_whole_period


def get_proportion_of_events(df, code, date):
    """
    Calculates the proportion of events for a given code on a given date.
    Args:
        df: A measure table.
        code: The code to calculate the proportion for.
        date: The date to calculate the proportion for.
    Returns:
        A dictionary containing the date, code and proportion.
    """
    total_events = df.loc[df["date"] == date, "num"].sum()
    code_events = df.loc[
        (df["date"] == date) & (df["code"] == code), "num"
    ].sum()
    row = {
        "date": date,
        "code": code,
    }
    if total_events == 0:
        row["proportion"] = 0
    else:
        row["proportion"] = (code_events / total_events) * 100
        row["count"] = code_events

    return row


def plot_top_codes_over_time(code_df, top_codes, measure, output_dir):
    """
    Plots the top 5 codes over time for each measure.
    Args:
        code_df: A codelist table.
        top_codes:A dictionary of the top codes for each measure with corresponding descriptions.
        measure: The measure ID.
        output_dir: The directory to save the plot to.

    """

    # Create a new dataframe with the proportion of events for each code on each date.
    code_proportions = pd.DataFrame()
    for code in top_codes.keys():
        code_proportions = pd.concat(
            [
                code_proportions,
                pd.DataFrame(
                    [
                        get_proportion_of_events(code_df, code, date)
                        for date in code_df["date"].unique()
                    ]
                ),
            ]
        )

    plt.figure(figsize=(10, 6))
    # seaborn styling
    sns.set_style("darkgrid")
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=colour_palette
    )
    plt.xlabel("Date")
    plt.ylabel("Count of codes")

    # Plot the proportion of events for each code on each date. Plots should be on the same graph.
    for code, description in top_codes.items():

        code_proportions.loc[code_proportions["code"] == code, :].plot(
            x="date", y="count", label=description, ax=plt.gca()
        )

    # legend outside of plot - top right
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.savefig(
        output_dir / f"{measure}_top_5_codes_over_time.png",
        bbox_inches="tight",
    )

    # save the underlying data for the plot
    code_proportions.to_csv(
        output_dir / f"{measure}_top_5_codes_over_time.csv", index=False
    )


def main():
    args = parse_args()
    input_file = args.input_file
    output_dir = args.output_dir

    measure_table = get_measure_tables(input_file)
    measure_table = coerce_numeric(measure_table)

    all_codes = {**MEDICATION_TO_CODELIST, **CLINICAL_TO_CODELIST}

    for key, codelist in all_codes.items():
        measure = f"event_code_{key}_rate"
        codelist = pd.read_csv(codelist, dtype="str")
        code_df = measure_table[measure_table["name"] == measure]

        code_column = "code"
        if "bnf_code" in codelist.columns and "dmd_name" in codelist.columns:
            term_column = "dmd_name"
            code_df = code_df.merge(
                codelist, left_on="group", right_on="dmd_id", how="left"
            )
            code_df = code_df.set_index("bnf_code")
            codelist = (
                codelist.groupby("bnf_code")[["dmd_name"]]
                .first()
                .reset_index()
            )
            codelist = codelist.rename(columns={"bnf_code": "code"})
            code_df = (
                code_df.groupby(["bnf_code", "date"])[["numerator"]]
                .sum()
                .reset_index()
            )
        else:

            term_column = "term"
            code_df = (
                code_df.groupby(["group", "date"])[["numerator"]]
                .sum()
                .reset_index()
            )

        # drop all columns except code and numerator
        code_df.columns = ["code", "date", "num"]
        
        (
            top_5_code_table_first,
            top_5_code_table_last,
            top_5_code_table,
        ) = create_top_5_code_table(
            df=code_df,
            code_df=codelist,
            code_column=code_column,
            term_column=term_column,
            low_count_threshold=5,
            rounding_base=10,
        )

        top_5_code_table.to_csv(
            output_dir / f"top_5_code_table_{measure}.csv", index=False
        )

        # find the top codes across the first and last date periods
        top_codes = list(
            set(top_5_code_table_first["Code"]).union(
                set(top_5_code_table_last["Code"])
            )
        )

        top_codes_dict = {
            i: codelist[codelist[code_column] == i][term_column].values[0]
            for i in top_codes
        }


        code_df["num"] = code_df["num"].apply(
            lambda x: round_values(x, base=10, redact=True, redaction_threshold=5)
        )
        
        # plot the top codes over time
        plot_top_codes_over_time(
            code_df=code_df,
            top_codes=top_codes_dict,
            measure=measure,
            output_dir=output_dir,
        )


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


if __name__ == "__main__":
    main()
