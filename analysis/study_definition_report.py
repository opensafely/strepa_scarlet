from cohortextractor import (
    StudyDefinition,
    patients,
    Measure,
    
)
from codelists import *

codelist_1 = allmed_review_codes
codelist_2 = dmard_codes

def generate_expectations_codes(codelist, incidence=0.5):
   
    expectations = {str(x): (1-incidence) / 10 for x in codelist[0:10]}
    # expectations = {str(x): (1-incidence) / len(codelist) for x in codelist}
    expectations[None] = incidence
    return expectations

start_date = "2019-01-01"
end_date = "2022-11-01"
# Specifiy study definition

demographics = {
    "age_band": (patients.categorised_as(
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

    )),
    "sex": (patients.sex(
        return_expectations={
            "rate": "universal",
            "category": {"ratios": {"M": 0.49, "F": 0.5, "U": 0.01}},
        }
    )),

    "region": (patients.registered_practice_as_of(
        "index_date",
        returning="nuts1_region_name",
        return_expectations={"category": {"ratios": {
            "North East": 0.1,
            "North West": 0.1,
            "Yorkshire and the Humber": 0.1,
            "East Midlands": 0.1,
            "West Midlands": 0.1,
            "East of England": 0.1,
            "London": 0.2,
            "South East": 0.2, }}}
    )),
    
    "imd": (patients.categorised_as(
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
    )),
        
}

# if these populations are selected, age band will already be extracted
# if population == "registered_children" or population == "registered_adults":
#     demographics.pop("age_band", None)

selected_demographics = {k: v for k, v in demographics.items() if k in breakdowns}

study = StudyDefinition(
    index_date="2019-01-01",
    default_expectations={
        "date": {"earliest": start_date, "latest": end_date},
        "rate": "exponential_increase",
        "incidence": 0.1,
    },
    
    population= selected_population,
    age=patients.age_as_of(
            "index_date",
            return_expectations={
                "rate": "universal",
                "int": {"distribution": "population_ages"},
            },
        ),
    **selected_demographics,
        
  
    practice=patients.registered_practice_as_of(
        "index_date",
        returning="pseudo_id",
        return_expectations={"int" : {"distribution": "normal", "mean": 25, "stddev": 5}, "incidence" : 0.5}
    ),
      
    event_1=patients.with_these_clinical_events(
        codelist=codelist_1,
        between=codelist_1_date_range,
        returning="binary_flag",
        return_expectations={"incidence": 0.5}
    ),

    event_1_code=patients.with_these_clinical_events(
        codelist=codelist_1,
        between=codelist_1_date_range,
        returning="code",
        return_expectations={
            "rate": "universal",
    population = patients.satisfying(
        """
        registered AND
        NOT died AND
        age >= 0 AND
        age <= 120
        """,

        registered = patients.registered_as_of(
            "index_date",
            return_expectations={"incidence": 0.9},
            ),

        died = patients.died_from_any_cause(
            on_or_before="index_date",
            returning="binary_flag",
            return_expectations={"incidence": 0.1}
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
        return_expectations={"int" : {"distribution": "normal", "mean": 25, "stddev": 5}, "incidence" : 0.5}
    ),


    
    
   
)

measures = [
    Measure(
        id=f"event_rate",
        numerator="event_measure",
        denominator="population",
        group_by=["practice"]
    ),
    Measure(
        id=f"event_code_1_rate",
        numerator="event_measure",
        denominator="population",
        group_by=["event_1_code"]
    ),
    Measure(
        id=f"event_code_2_rate",
        numerator="event_measure",
        denominator="population",
        group_by=["event_2_code"]
    ),
]

if breakdowns:
    for b in breakdowns:
        measures.append(
            Measure(
                id=f"event_{b}_rate",
                numerator="event_measure",
                denominator="population",
                group_by=[b],
            ),
        )


   