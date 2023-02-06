from cohortextractor import (
    StudyDefinition,
    patients,
    Measure,
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
    strep_a_sore_throat_codes,
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
    "strep_a_sore_throat": strep_a_sore_throat_codes,
}


ENDDATE = "index_date + 6 days"


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


def generate_expectations_codes(codelist, incidence=0.5):

    expectations = {str(x): (1 - incidence) / 10 for x in codelist[0:10]}
    expectations[None] = incidence
    return expectations


start_date = "2018-01-01"
end_date = "2022-12-01"
# Specifiy study definition

demographics = {
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
                        "0-4": 0.3,
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
    "region": (
        patients.registered_practice_as_of(
            "index_date",
            returning="nuts1_region_name",
            return_expectations={
                "category": {
                    "ratios": {
                        "North East": 0.1,
                        "North West": 0.1,
                        "Yorkshire and the Humber": 0.1,
                        "East Midlands": 0.1,
                        "West Midlands": 0.1,
                        "East of England": 0.1,
                        "London": 0.2,
                        "South East": 0.2,
                    }
                }
            },
        )
    ),
    "imd": (
        patients.categorised_as(
            {
                "missing": "DEFAULT",
                "1 - most deprived": """index_of_multiple_deprivation >=0 AND index_of_multiple_deprivation < 32844*1/5""",
                "2": """index_of_multiple_deprivation >= 32844*1/5 AND index_of_multiple_deprivation < 32844*2/5""",
                "3": """index_of_multiple_deprivation >= 32844*2/5 AND index_of_multiple_deprivation < 32844*3/5""",
                "4": """index_of_multiple_deprivation >= 32844*3/5 AND index_of_multiple_deprivation < 32844*4/5""",
                "5 - least deprived": """index_of_multiple_deprivation >= 32844*4/5 AND index_of_multiple_deprivation < 32844""",
            },
            index_of_multiple_deprivation=patients.address_as_of(
                "index_date",
                returning="index_of_multiple_deprivation",
                round_to_nearest=100,
            ),
            return_expectations={
                "rate": "universal",
                "category": {
                    "ratios": {
                        "missing": 0.05,
                        "1 - most deprived": 0.19,
                        "2": 0.19,
                        "3": 0.19,
                        "4": 0.19,
                        "5 - least deprived": 0.19,
                    }
                },
            },
        )
    ),
    "practice": patients.registered_practice_as_of(
        "index_date",
        returning="pseudo_id",
        return_expectations={
            "int": {"distribution": "normal", "mean": 25, "stddev": 5},
            "incidence": 0.9,
        },
    ),
}


clinical_events = [
    {
        f"event_{clinical_key}": patients.with_these_clinical_events(
            codelist=clinical_codelist,
            between=["index_date", ENDDATE],
            returning="binary_flag",
            include_date_of_match=True,
            date_format="YYYY-MM",
            return_expectations={"incidence": 0.1},
        ),
        f"event_code_{clinical_key}": patients.with_these_clinical_events(
            codelist=clinical_codelist,
            between=["index_date", ENDDATE],
            returning="code",
            return_expectations={
                "rate": "universal",
                "category": {
                    "ratios": generate_expectations_codes(clinical_codelist)
                },
            },
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
            include_date_of_match=True,
            date_format="YYYY-MM",
            return_expectations={"incidence": 0.1},
        ),
        f"event_code_{medication_key}": patients.with_these_medications(
            codelist=medication_codelist,
            between=["index_date", ENDDATE],
            returning="code",
            return_expectations={
                "rate": "universal",
                "category": {
                    "ratios": generate_expectations_codes(medication_codelist)
                },
            },
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
    population=patients.satisfying(
        """
        registered AND
        NOT died AND
        age >= 0 AND
        age < 120 AND
        age_band != "missing"
        """,
        registered=patients.registered_as_of(
            "index_date",
            return_expectations={"incidence": 0.9},
        ),
        died=patients.died_from_any_cause(
            on_or_before="index_date",
            returning="binary_flag",
            return_expectations={"incidence": 0.1},
        ),
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
)

# Ethnicity isn't in the demographics dict because it's extracted in a separate
# study definition. We add it here because we want to calculate a measure using
# it. We only care about the key.

measures = []
# add measure for each codelist
for medication_key in list(medication_codelists.keys()):
    measures.extend(
        [
            Measure(
                id=f"event_{medication_key}_rate",
                numerator=f"event_{medication_key}",
                denominator="population",
                group_by=["population"],
                small_number_suppression=True,
            ),
            Measure(
                id=f"event_code_{medication_key}_rate",
                numerator=f"event_{medication_key}",
                denominator="population",
                group_by=[f"event_code_{medication_key}"],
                small_number_suppression=True,
            ),
        ]
    )

    for d in demographics.keys():
        if d == "practice":
            suppress = False
        else:
            suppress = True
        measures.extend(
            [
                Measure(
                    id=f"event_{medication_key}_{d}_rate",
                    numerator=f"event_{medication_key}",
                    denominator="population",
                    group_by=[d],
                    small_number_suppression=suppress,
                ),
            ]
        )

for clinical_key in list(clinical_event_codelists.keys()):
    measures.extend(
        [
            Measure(
                id=f"event_{clinical_key}_rate",
                numerator=f"event_{clinical_key}",
                denominator="population",
                group_by=["population"],
                small_number_suppression=True,
            ),
            Measure(
                id=f"event_code_{clinical_key}_rate",
                numerator=f"event_{clinical_key}",
                denominator="population",
                group_by=[f"event_code_{clinical_key}"],
                small_number_suppression=True,
            ),
        ]
    )

    for d in demographics.keys():
        if d == "practice":
            suppress = False
        else:
            suppress = True
        measures.extend(
            [
                Measure(
                    id=f"event_{clinical_key}_{d}_rate",
                    numerator=f"event_{clinical_key}",
                    denominator="population",
                    group_by=[d],
                    small_number_suppression=suppress,
                ),
            ]
        )

# Medication any and clinical any
measures.extend(
    [
        Measure(
            id="event_medication_any_rate",
            numerator="event_medication_any",
            denominator="population",
            group_by=["population"],
            small_number_suppression=True,
        ),
        Measure(
            id="event_clinical_any_rate",
            numerator="event_clinical_any",
            denominator="population",
            group_by=["population"],
            small_number_suppression=True,
        ),
    ]
)

for d in demographics.keys():
    if d == "practice":
        suppress = False
    else:
        suppress = True
    measures.extend(
        [
            Measure(
                id=f"event_medication_any_{d}_rate",
                numerator="event_medication_any",
                denominator="population",
                group_by=[d],
                small_number_suppression=suppress,
            ),
            Measure(
                id=f"event_clinical_any_{d}_rate",
                numerator="event_clinical_any",
                denominator="population",
                group_by=[d],
                small_number_suppression=suppress,
            ),
        ]
    )
