from cohortextractor import (
    StudyDefinition,
    patients,
)

from codelists import (
    sore_throat_prediction_codes,
    throat_swab_codes,
)


def generate_expectations_codes(codelist, incidence=0.5):
    expectations = {str(x): (1 - incidence) / len(codelist) for x in codelist}
    expectations[None] = incidence
    return expectations


start_date = "2019-01-01"
end_date = "2022-01-01"

study = StudyDefinition(
    index_date=start_date,
    # Configure the expectations framework
    default_expectations={
        "date": {"earliest": start_date, "latest": end_date},
        "rate": "uniform",
        "incidence": 0.1,
    },
    # Define the study population
    population=patients.all(),
    age=patients.age_as_of(
        "index_date",
        return_expectations={
            "rate": "universal",
            "int": {"distribution": "population_ages"},
        },
    ),
    throat_swab=patients.with_these_clinical_events(
        codelist=throat_swab_codes,
        between=["index_date", "last_day_of_month(index_date)"],
        returning="binary_flag",
        date_format="YYYY-MM-DD",
        include_date_of_match=True,
        return_expectations={
            "incidence": 0.5,
            "date": {"earliest": "1900-01-01", "latest": "today"},
        },
    ),
    throat_swab_code=patients.with_these_clinical_events(
        codelist=throat_swab_codes,
        between=["index_date", "last_day_of_month(index_date)"],
        returning="code",
        return_expectations={
            "rate": "universal",
            "category": {
                "ratios": generate_expectations_codes(throat_swab_codes)
            },
        },
    ),
    throat_swab_count=patients.with_these_clinical_events(
        codelist=throat_swab_codes,
        between=["index_date", "last_day_of_month(index_date)"],
        returning="number_of_matches_in_period",
        return_expectations={
            "int": {"distribution": "poisson", "mean": 2},
            "incidence": 0.2,
        },
    ),
    sore_throat_prediction=patients.with_these_clinical_events(
        codelist=sore_throat_prediction_codes,
        between=["index_date", "last_day_of_month(index_date)"],
        returning="binary_flag",
        date_format="YYYY-MM-DD",
        include_date_of_match=True,
        return_expectations={
            "incidence": 0.5,
            "date": {"earliest": "1900-01-01", "latest": "today"},
        },
    ),
    sore_throat_prediction_code=patients.with_these_clinical_events(
        codelist=sore_throat_prediction_codes,
        between=["index_date", "last_day_of_month(index_date)"],
        returning="code",
        return_expectations={
            "rate": "universal",
            "category": {
                "ratios": generate_expectations_codes(
                    sore_throat_prediction_codes
                )
            },
        },
    ),
    sore_throat_prediction_count=patients.with_these_clinical_events(
        codelist=sore_throat_prediction_codes,
        between=["index_date", "last_day_of_month(index_date)"],
        returning="number_of_matches_in_period",
        return_expectations={
            "int": {"distribution": "poisson", "mean": 2},
            "incidence": 0.2,
        },
    ),
    sore_throat_prediction_value=patients.with_these_clinical_events(
        codelist=sore_throat_prediction_codes,
        between=["index_date", "last_day_of_month(index_date)"],
        returning="numeric_value",
        return_expectations={
            "float": {"distribution": "normal", "mean": 45.0, "stddev": 20},
            "incidence": 0.5,
        },
    ),
    sore_throat_prediction_operator=patients.comparator_from(
        "sore_throat_prediction_value",
        return_expectations={
            "rate": "universal",
            "category": {
                "ratios": {  # ~, =, >= , > , < , <=
                    None: 0.10,
                    "~": 0.05,
                    "=": 0.65,
                    ">=": 0.05,
                    ">": 0.05,
                    "<": 0.05,
                    "<=": 0.05,
                }
            },
            "incidence": 0.80,
        },
    ),
)
