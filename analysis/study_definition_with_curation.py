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
    scarlet_fever_codes,
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
    population=patients.satisfying(
        """
    scarlet_fever
    """,
    ),
    age=patients.age_as_of(
        "last_day_of_month(index_date) + 1 day",
        return_expectations={
            "rate": "universal",
            "int": {"distribution": "population_ages"},
        },
    ),
    scarlet_fever=patients.with_these_clinical_events(
        codelist=scarlet_fever_codes,
        between=["index_date", "last_day_of_month(index_date)"],
        returning="binary_flag",
        include_date_of_match=True,
        date_format="YYYY-MM",
        return_expectations={
            "incidence": 0.5,
            "date": {"earliest": "1900-01-01", "latest": "today"},
        },
    ),
    any_prescription=patients.with_these_medications(
        codelist=any_medication_codes,
        between=["index_date", "last_day_of_month(index_date)"],
        returning="binary_flag",
        include_date_of_match=True,
        return_expectations={
            "incidence": 0.5,
            "date": {"earliest": "1900-01-01", "latest": "today"},
        },
    ),
    any_prescription_count=patients.with_these_medications(
        codelist=any_medication_codes,
        between=["index_date", "last_day_of_month(index_date)"],
        returning="number_of_matches_in_period",
        return_expectations={
            "int": {"distribution": "normal", "mean": 2, "stddev": 1},
            "incidence": 0.5,
        },
    ),
    any_prescription_code=patients.with_these_medications(
        codelist=any_medication_codes,
        between=["index_date", "last_day_of_month(index_date)"],
        returning="code",
        return_expectations={
            "rate": "universal",
            "category": {
                "ratios": generate_expectations_codes(any_medication_codes)
            },
        },
    ),
    scarlet_fever_with_medication_any=patients.with_these_medications(
        codelist=any_medication_codes,
        between=[
            "scarlet_fever_date - 7 days",
            "scarlet_fever_date + 14 days",
        ],
        returning="binary_flag",
    ),
    scarlet_fever_with_medication_any_same_day=patients.with_these_medications(
        codelist=any_medication_codes,
        between=[
            "scarlet_fever_date",
            "scarlet_fever_date",
        ],
        returning="binary_flag",
    ),
)
