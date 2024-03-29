import pathlib
import argparse
import pandas
from report_utils import (
    ci_95_proportion,
    ci_to_str,
    get_measure_tables,
    is_bool_as_int,
    match_paths,
    reorder_dashes,
)

"""
Generate table1 from joined measures file.
Group by category and group, and then further split into columns based on user
provided column names
"""


def parse_column(columns):
    if ":" in columns:
        return tuple(columns.split(":"))
    else:
        return (columns, columns.replace("_", " ").title())


# NOTE: this is slightly different than the utils subset_table
# because it takes a list of patterns
# TODO: standardise to one subset_table
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


def series_to_bool(series):
    if is_bool_as_int(series):
        return series.astype(int).astype(bool)
    else:
        return series


def transform_percentage(x):
    """
    Return (comma delimited) x and column percentage
    """
    transformed = (
        x.map("{:,.0f}".format)
        + " ("
        + (((x / x.sum()) * 100).round(1)).astype(str)
        + ")"
    )
    transformed.name = f"{x.name} (%)"
    return transformed


def get_percentages(df, include_denominator, exclude_numerator):
    """
    Create a new column which has count (%) of group
    After computation is complete reconvert numeric to string and replace
    nan with "REDACTED" again
    """

    percent = df.groupby(level=0).transform(transform_percentage)
    percent = percent.replace("nan (nan)", "[REDACTED]")

    if include_denominator:
        cis = ci_95_proportion(df, scale=1000)
        rate = ci_to_str(cis)
        rate = rate.replace("nan (nan to nan)", "[REDACTED]")
        percent["rate"] = rate
    else:
        percent = percent.drop("denominator", axis=1)
    if exclude_numerator:
        percent = percent.drop("numerator", axis=1)
    percent = percent.rename(
        columns={
            "numerator": "No. with event (%)",
            "denominator": "No. registered patients (%)",
            "rate": "Rate per 1,000 (95% CI)",
        }
    )
    return percent


def remove_duplicate_cols(df):
    denoms = df.loc[:, (slice(None), "No. registered patients (%)")]
    equal = denoms.eq(denoms.iloc[:, 0], axis=0).all(1)
    skip_total = equal.drop(("Total", ""))
    if skip_total.all():
        shared_denom = denoms[denoms.columns[0]]
        shared_denom.name = (
            "Overall Population",
            "No. registered patients (%)",
        )
        dropped_denom = df.drop(denoms.columns, axis=1)
        return pandas.concat([shared_denom, dropped_denom], axis=1)
    return df


def title_multiindex(df):
    titled = []
    # NOTE: dataframe must be sorted, otherwise new index may not match
    df = df.sort_index()
    for category, data in df.groupby(level=0):
        category = category.replace("_", " ")
        group = data.index.get_level_values(1).to_series()
        group = group.replace({"Unknown": "Missing"})
        group = group.fillna("Missing")
        group = series_to_bool(group)
        titled += [
            (str(category).title(), str(item).title())
            for item in group.to_list()
        ]
    df.index = pandas.MultiIndex.from_tuples(titled)
    return df


def reorder_dataframe(df):
    # Use a dataframe to preserve order
    # TODO: investigate use of sort_values
    reordered = pandas.DataFrame()
    for category, data in df.groupby(level=0):
        level_1_values = data.index.get_level_values(1)
        data = reorder_dashes(data)
        if "Missing" in level_1_values:
            reordered = pandas.concat(
                [
                    data.drop(labels=(category, "Missing")).append(
                        data.loc[(category, "Missing")]
                    ),
                    reordered,
                ]
            )
        else:
            reordered = pandas.concat([data, reordered])
    return reordered


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to single joined measures file",
    )
    parser.add_argument(
        "--measures-pattern",
        required=True,
        action="append",
        help="Glob pattern matching one or more measures names for rows",
    )
    parser.add_argument(
        "--column-names",
        nargs="+",
        type=parse_column,
        help="Split measures with these names into separate columns",
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
        "--include-denominator",
        action="store_true",
        help="Include denominator (%) and rate",
    )
    parser.add_argument(
        "--exclude-numerator",
        action="store_true",
        help="Exclude numerator (%)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input_file
    measures_pattern = args.measures_pattern
    output_dir = args.output_dir
    output_name = args.output_name
    columns = args.column_names
    include_denominator = args.include_denominator
    exclude_numerator = args.exclude_numerator

    output_dir.mkdir(parents=True, exist_ok=True)

    measure_table = get_measure_tables(input_file)

    end_date = measure_table.date.max()
    subset = subset_table(measure_table, measures_pattern, end_date)

    table1 = pandas.DataFrame()
    for column, column_title in columns:
        sub = subset[subset.name.str.contains(column)]
        sub = sub.set_index(["category", "group"])
        sub = sub[["numerator", "denominator"]]
        # NOTE: may not be true total, could pull from total measure
        overall = sub.loc[sub.iloc[0].name[0]].sum()
        overall.name = ("Total", "")
        sub = pandas.concat([pandas.DataFrame(overall).T, sub])
        sub = get_percentages(sub, include_denominator, exclude_numerator)
        sub.columns = pandas.MultiIndex.from_product(
            [[column_title], sub.columns]
        )
        if table1.empty:
            table1 = sub
        else:
            table1 = table1.join(sub)

    table1 = remove_duplicate_cols(table1)

    table1 = title_multiindex(table1)
    table1 = reorder_dataframe(table1)
    table1.to_html(output_dir / output_name, index=True)


if __name__ == "__main__":
    main()
