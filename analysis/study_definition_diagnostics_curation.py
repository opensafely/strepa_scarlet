from cohortextractor import StudyDefinition, patients, combine_codelists

from codelists import (
    amoxicillin_codes,
    azithromycin_codes,
    clarithromycin_codes,
    erythromycin_codes,
    phenoxymethylpenicillin_codes,
    cefalexin_codes,
    co_amoxiclav_codes,
    flucloxacillin_codes,
    strep_a_sore_throat_codes,
    sore_throat_prediction_codes,
    throat_swab_codes,
)

any_medication_codes = combine_codelists(
    amoxicillin_codes,
    azithromycin_codes,
    clarithromycin_codes,
    erythromycin_codes,
    phenoxymethylpenicillin_codes,
    cefalexin_codes,
    co_amoxiclav_codes,
    flucloxacillin_codes,
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
        "last_day_of_month(index_date) + 1 day",
        return_expectations={
            "rate": "universal",
            "int": {"distribution": "population_ages"},
        },
    ),
    any_prescription=patients.with_these_clinical_events(
        codelist=any_medication_codes,
        between=["index_date", "last_day_of_month(index_date)"],
        include_date_of_match=True,
        return_expectations={
            "incidence": 0.5,
            "date": {"earliest": "1900-01-01", "latest": "today"},
        },
    ),
    strep_a_sore_throat=patients.with_these_clinical_events(
        codelist=strep_a_sore_throat_codes,
        between=["index_date", "last_day_of_month(index_date)"],
        returning="binary_flag",
        return_expectations={
            "incidence": 0.5,
            "date": {"earliest": "1900-01-01", "latest": "today"},
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
    sore_throat_score_and_antibiotic=patients.categorised_as(
        {
            "missing": "DEFAULT",
            "0": "sore_throat_prediction_value='0' AND any_prescription",
            "1": "sore_throat_prediction_value='1' AND any_prescription",
            "2": "sore_throat_prediction_value='2' AND any_prescription",
            "3": "sore_throat_prediction_value='3' AND any_prescription",
            "4": "sore_throat_prediction_value='4' AND any_prescription",
            "5": "sore_throat_prediction_value='5' AND any_prescription",
        },
        return_expectations={
            "rate": "universal",
            "category": {
                "ratios": {
                    "missing": 0.25,
                    "0": 0.125,
                    "1": 0.125,
                    "2": 0.125,
                    "3": 0.125,
                    "4": 0.125,
                    "5": 0.125,
                }
            },
        },
    ),
    fever_pain_diagnosis=patients.categorised_as(
        {
            "missing": "DEFAULT",
            "sore_throat_fever_pain_prescription": "any_prescription AND strep_a_sore_throat AND sore_throat_prediction",
            "sore_throat_throat_swab_prescription": "any_prescription AND strep_a_sore_throat AND throat_swab",
            "fever_pain_prescription": "any_prescription AND sore_throat_prediction",
            "throat_swab_prescription": "any_prescription AND throat_swab",
            "prescription": "any_prescription",
        },
        return_expectations={
            "rate": "universal",
            "category": {
                "ratios": {
                    "missing": 0.30,
                    "sore_throat_fever_pain_prescription": 0.10,
                    "sore_throat_throat_swab_prescription": 0.10,
                    "fever_pain_prescription": 0.2,
                    "throat_swab_prescription": 0.2,
                    "prescription": 0.1,
                }
            },
        },
    ),
)
