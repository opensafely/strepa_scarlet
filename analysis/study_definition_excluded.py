from cohortextractor import (
    StudyDefinition,
    patients,
    params,
    combine_codelists,
)

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
    invasive_strep_a_codes,
    sore_throat_tonsillitis_codes,
)

# Import so we can inspect metadata logs for correct variable expansion
import logging
import json


medication_codelists = {
    "amoxicillin": amoxicillin_codes,
    "azithromycin": azithromycin_codes,
    "clarithromycin": clarithromycin_codes,
    "erythromycin": erythromycin_codes,
    "phenoxymethylpenicillin": phenoxymethylpenicillin_codes,
    "cefalexin": cefalexin_codes,
    "co_amoxiclav": co_amoxiclav_codes,
    "flucloxacillin": flucloxacillin_codes,
}


clinical_event_codelists = {
    "scarlet_fever": scarlet_fever_codes,
    "invasive_strep_a": invasive_strep_a_codes,
    "sore_throat_tonsillitis": sore_throat_tonsillitis_codes,
}

all_medication_codes = combine_codelists(*list(medication_codelists.values()))
all_clinical_codes = combine_codelists(
    *list(clinical_event_codelists.values())
)


frequency = params.get("frequency", None)
if frequency == "weekly":
    ENDDATE = "index_date + 6 days"
else:
    ENDDATE = "last_day_of_month(index_date)"


def generate_all_medications():
    var = {
        "event_medication_any": patients.satisfying(
            " OR ".join(
                list(map(lambda x: f"event_{x}", medication_codelists.keys()))
            )
        ),
    }
    logging.info(json.dumps(var, indent=4))
    return var


def generate_all_clinical():
    var = {
        "event_clinical_any": patients.satisfying(
            " OR ".join(
                list(
                    map(
                        lambda x: f"event_{x}", clinical_event_codelists.keys()
                    )
                )
            )
        ),
    }
    logging.info(json.dumps(var, indent=4))
    return var


if frequency == "weekly":
    start_date = "2022-09-01"
    end_date = "2023-02-15"
else:
    start_date = "2018-01-01"
    end_date = "2022-01-01"

demographics = {
    "sex": patients.sex(
        return_expectations={
            "rate": "universal",
            "category": {"ratios": {"M": 0.49, "F": 0.50, "U": 0.01}},
        }
    ),
    "age_band": (
        patients.categorised_as(
            {
                "missing": "DEFAULT",
                "0-4": """ age >=  0 AND age < 5""",
                "5-9": """ age >=  5 AND age < 10""",
                "10-14": """ age >=  10 AND age < 15""",
                "15-44": """ age >=  15 AND age < 45""",
                "45-64": """ age >=  45 AND age < 65""",
                "65-74": """ age >=  65 AND age < 75""",
                "75+": """ age >=  75 AND age < 120""",
            },
            return_expectations={
                "rate": "universal",
                "category": {
                    "ratios": {
                        "missing": 0.05,
                        "0-4": 0.25,
                        "5-9": 0.3,
                        "10-14": 0.1,
                        "15-44": 0.1,
                        "45-64": 0.1,
                        "75+": 0.1,
                    }
                },
            },
        )
    ),
}


clinical_events = [
    {
        f"event_{clinical_key}": patients.with_these_clinical_events(
            codelist=clinical_codelist,
            between=["index_date", ENDDATE],
            returning="binary_flag",
            return_expectations={"incidence": 0.1},
        ),
    }
    for clinical_key, clinical_codelist in clinical_event_codelists.items()
]


medication_events = [
    {
        f"event_{medication_key}": patients.with_these_medications(
            codelist=medication_codelist,
            between=["index_date", ENDDATE],
            returning="binary_flag",
            return_expectations={"incidence": 0.1},
        ),
    }
    for medication_key, medication_codelist in medication_codelists.items()
]
# convert list of dicts into a single dict
medication_variables = {k: v for d in medication_events for k, v in d.items()}
clinical_event_variables = {
    k: v for d in clinical_events for k, v in d.items()
}

study = StudyDefinition(
    index_date="2019-01-01",
    default_expectations={
        "date": {"earliest": start_date, "latest": end_date},
        "rate": "exponential_increase",
        "incidence": 0.1,
    },
    population=patients.all(),
    registered=patients.registered_as_of(
        "index_date",
        return_expectations={"incidence": 0.9},
    ),
    died=patients.died_from_any_cause(
        on_or_before="index_date",
        returning="binary_flag",
        return_expectations={"incidence": 0.1},
    ),
    age=patients.age_as_of(
        "index_date",
        return_expectations={
            "rate": "universal",
            "int": {"distribution": "population_ages"},
        },
    ),
    **demographics,
    **clinical_event_variables,
    **medication_variables,
    **generate_all_medications(),
    **generate_all_clinical(),
    included=patients.satisfying(
        """
        registered AND
        NOT died AND
        age_band != "missing" AND
        (sex = "M" OR sex = "F")
        """
    ),
)
