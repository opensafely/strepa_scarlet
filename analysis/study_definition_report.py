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


def generate_all_medications():
    return {
        "medication_any": patients.satisfying(
            " OR ".join(
                list(map(lambda x: f"event_{x}", medication_codelists.keys()))
            )
        )
    }


def generate_expectations_codes(codelist, incidence=0.5):

    expectations = {str(x): (1 - incidence) / 10 for x in codelist[0:10]}
    # expectations = {str(x): (1-incidence) / len(codelist) for x in codelist}
    expectations[None] = incidence
    return expectations


start_date = "2019-01-01"
end_date = "2022-11-01"
# Specifiy study definition

demographics = {
    "age_band": (
        patients.categorised_as(
            {
                "missing": "DEFAULT",
                "18-19": """ age >= 0 AND age < 20""",
                "20-29": """ age >=  20 AND age < 30""",
                "30-39": """ age >=  30 AND age < 40""",
                "40-49": """ age >=  40 AND age < 50""",
                "50-59": """ age >=  50 AND age < 60""",
                "60-69": """ age >=  60 AND age < 70""",
                "70-79": """ age >=  70 AND age < 80""",
                "80+": """ age >=  80 AND age < 120""",
            },
            return_expectations={
                "rate": "universal",
                "category": {
                    "ratios": {
                        "missing": 0.005,
                        "18-19": 0.125,
                        "20-29": 0.125,
                        "30-39": 0.125,
                        "40-49": 0.125,
                        "50-59": 0.125,
                        "60-69": 0.125,
                        "70-79": 0.125,
                        "80+": 0.12,
                    }
                },
            },
        )
    ),
    "sex": (
        patients.sex(
            return_expectations={
                "rate": "universal",
                "category": {"ratios": {"M": 0.49, "F": 0.5, "U": 0.01}},
            }
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
        f"event_{k}": patients.with_these_medications(
            codelist=c,
            between=["index_date", "last_day_of_month(index_date)"],
            returning="binary_flag",
            include_date_of_match=True,
            date_format="YYYY-MM",
            return_expectations={"incidence": 0.1},
        ),
        f"event_code_{k}": patients.with_these_medications(
            codelist=c,
            between=["index_date", "last_day_of_month(index_date)"],
            returning="code",
            return_expectations={
                "rate": "universal",
                "category": {"ratios": generate_expectations_codes(c)},
            },
        ),
    }
    for k, c in medication_codelists.items()
]

clinical_events = [
    {
        f"event_{k}": patients.with_these_clinical_events(
            codelist=c,
            between=["index_date", "last_day_of_month(index_date)"],
            returning="binary_flag",
            include_date_of_match=True,
            date_format="YYYY-MM",
            return_expectations={"incidence": 0.1},
        ),
        f"event_code_{k}": patients.with_these_clinical_events(
            codelist=c,
            between=["index_date", "last_day_of_month(index_date)"],
            returning="code",
            return_expectations={
                "rate": "universal",
                "category": {"ratios": generate_expectations_codes(c)},
            },
        ),
    }
    for k, c in clinical_event_codelists.items()
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
        age <= 120
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
    **medication_variables,
    **clinical_event_variables,
    **generate_all_medications(),
)

measures = []

# add measure for each codelist
for k, c in medication_codelists.items():
    measures.extend(
        [
            Measure(
                id=f"event_{k}_rate",
                numerator=f"event_{k}",
                denominator="population",
                group_by=["population"],
            ),
            Measure(
                id=f"event_code_{k}_rate",
                numerator=f"event_{k}",
                denominator="population",
                group_by=[f"event_code_{k}"],
            ),
        ]
    )

    for d in demographics.keys():
        measures.append(
            Measure(
                id=f"event_{k}_{d}_rate",
                numerator=f"event_{k}",
                denominator="population",
                group_by=[d],
            ),
        )

for k, c in clinical_event_codelists.items():
    measures.extend(
        [
            Measure(
                id=f"event_{k}_rate",
                numerator=f"event_{k}",
                denominator="population",
                group_by=["population"],
            ),
            Measure(
                id=f"event_code_{k}_rate",
                numerator=f"event_{k}",
                denominator="population",
                group_by=[f"event_code_{k}"],
            ),
        ]
    )

    for d in demographics.keys():
        measures.append(
            Measure(
                id=f"event_{k}_{d}_rate",
                numerator=f"event_{k}",
                denominator="population",
                group_by=[d],
            ),
        )