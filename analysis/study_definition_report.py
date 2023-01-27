from cohortextractor import (
    StudyDefinition,
    patients,
    Measure,
    params,
)

from codelists import (
    amoxicillin_codes,
    azithromycin_codes,
    clarithromycin_codes,
    erythromycin_codes,
    phenoxymethypenicillin_codes,
    scarlet_fever_codes,
    invasive_strep_a_codes,
    strep_a_sore_throat_codes,
)

medication_codelists = {
    "amoxicillin": amoxicillin_codes,
    "azithromycin": azithromycin_codes,
    "clarithromycin": clarithromycin_codes,
    "erythromycin": erythromycin_codes,
    "phenoxymethypenicillin": phenoxymethypenicillin_codes,
}


clinical_event_codelists = {
    "scarlet_fever": scarlet_fever_codes,
    "invasive_strep_a": invasive_strep_a_codes,
    "strep_a_sore_throat": strep_a_sore_throat_codes,
}


frequency = params.get("frequency", None)
if frequency == "weekly":
    ENDDATE = "index_date + 6 days"
else:
    ENDDATE = "last_day_of_month(index_date)"


def generate_all_medications():
    return {
        "event_medication_any": patients.satisfying(
            " OR ".join(
                list(map(lambda x: f"event_{x}", medication_codelists.keys()))
            )
        ),
    }


def generate_all_medications_2_weeks(clinical_events_codelists):
    return {
        f"{clinical_key}_medication_any_2_weeks": patients.satisfying(
            " OR ".join(
                list(
                    map(
                        lambda medication_key: f"event_{clinical_key}_with_{medication_key}",
                        medication_codelists.keys(),
                    )
                )
            )
        )
        for clinical_key in clinical_events_codelists.keys()
    }


def generate_expectations_codes(codelist, incidence=0.5):

    expectations = {str(x): (1 - incidence) / 10 for x in codelist[0:10]}
    # expectations = {str(x): (1-incidence) / len(codelist) for x in codelist}
    expectations[None] = incidence
    return expectations


start_date = "2019-01-01"
end_date = "2022-12-01"
# Specifiy study definition

demographics = {
    "age_band": (
        patients.categorised_as(
            {
                "missing": "DEFAULT",
                "Under 1": """ age >= 0 AND age <1""",
                "1-4": """ age >=  1 AND age < 5""",
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
                        "Under 1": 0.15,
                        "1-4": 0.3,
                        "5-9": 0.3,
                        "10-14": 0.05,
                        "15-44": 0.05,
                        "45-64": 0.05,
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
                "0": "DEFAULT",
                "1": """index_of_multiple_deprivation >=0 AND index_of_multiple_deprivation < 32844*1/5""",
                "2": """index_of_multiple_deprivation >= 32844*1/5 AND index_of_multiple_deprivation < 32844*2/5""",
                "3": """index_of_multiple_deprivation >= 32844*2/5 AND index_of_multiple_deprivation < 32844*3/5""",
                "4": """index_of_multiple_deprivation >= 32844*3/5 AND index_of_multiple_deprivation < 32844*4/5""",
                "5": """index_of_multiple_deprivation >= 32844*4/5 AND index_of_multiple_deprivation < 32844""",
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
                        "0": 0.05,
                        "1": 0.19,
                        "2": 0.19,
                        "3": 0.19,
                        "4": 0.19,
                        "5": 0.19,
                    }
                },
            },
        )
    ),
}


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
        **{
            f"event_{clinical_key}_with_{medication_key}": patients.with_these_medications(
                codelist=medication_codelist,
                between=[
                    f"event_{clinical_key}_date - 7 days",
                    f"event_{clinical_key}_date + 14 days",
                ],
                returning="binary_flag",
            )
            for clinical_key in clinical_event_codelists.keys()
        },
    }
    for medication_key, medication_codelist in medication_codelists.items()
]

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
    # keep this in case we do want it later
    practice=patients.registered_practice_as_of(
        "index_date",
        returning="pseudo_id",
        return_expectations={
            "int": {"distribution": "normal", "mean": 25, "stddev": 5},
            "incidence": 0.5,
        },
    ),
    **clinical_event_variables,
    **medication_variables,
    **generate_all_medications(),
    **generate_all_medications_2_weeks(clinical_event_codelists),
)

# Ethnicity isn't in the demographics dict because it's extracted in a separate
# study definition. We add it here because we want to calculate a measure using it.
# We only care about the key.

demographics["ethnicity"] = ""

measures = []

# add measure for each codelist
for medication_key in medication_codelists.keys():
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
        measures.append(
            Measure(
                id=f"event_{medication_key}_{d}_rate",
                numerator=f"event_{medication_key}",
                denominator="population",
                group_by=[d],
                small_number_suppression=True,
            ),
        )

# Any medication
measures.append(
    Measure(
        id="event_medication_any_rate",
        numerator="event_medication_any",
        denominator="population",
        group_by=["population"],
        small_number_suppression=True,
    )
)
for d in demographics.keys():
    measures.append(
        Measure(
            id=f"event_medication_any_{d}_rate",
            numerator="event_medication_any",
            denominator="population",
            group_by=[d],
            small_number_suppression=True,
        ),
    )

for clinical_key in clinical_event_codelists.keys():
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
            Measure(
                id=f"event_{clinical_key}_medication_any_2_weeks_rate",
                numerator=f"{clinical_key}_medication_any_2_weeks",
                denominator="population",
                group_by=["population"],
                small_number_suppression=True,
            ),
        ]
    )

    for d in demographics.keys():
        measures.extend([
            Measure(
                id=f"event_{clinical_key}_{d}_rate",
                numerator=f"event_{clinical_key}",
                denominator="population",
                group_by=[d],
                small_number_suppression=True,
            ),
            Measure(
                id=f"event_{clinical_key}_medication_any_2_weeks_{d}_rate",
                numerator=f"{clinical_key}_medication_any_2_weeks",
                denominator="population",
                group_by=[d],
                small_number_suppression=True,
            ),
            ]
        )
