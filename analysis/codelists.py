from cohortextractor import (
    codelist_from_csv,
)

ethnicity_codes = codelist_from_csv(
    "codelists/opensafely-ethnicity-snomed-0removed.csv",
    system="snomed",
    column="snomedcode",
    category_column="Grouping_6",
)

amoxicillin_codes = codelist_from_csv(
    "codelists/opensafely-amoxicillin-oral.csv",
    system="snomed",
    column="dmd_id",
)
azithromycin_codes = codelist_from_csv(
    "codelists/opensafely-azithromycin-oral.csv",
    system="snomed",
    column="dmd_id",
)
clarithromycin_codes = codelist_from_csv(
    "codelists/opensafely-clarithromycin-oral.csv",
    system="snomed",
    column="dmd_id",
)
erythromycin_codes = codelist_from_csv(
    "codelists/opensafely-erythromycin-oral.csv",
    system="snomed",
    column="dmd_id",
)
phenoxymethypenicillin_codes = codelist_from_csv(
    "codelists/opensafely-phenoxymethypenicillin.csv",
    system="snomed",
    column="dmd_id",
)

scarlet_fever_codes = codelist_from_csv(
    "codelists/user-chriswood-scarlet-fever.csv",
    system="snomed",
    column="code",
)

invasive_strep_a_codes = codelist_from_csv(
    "codelists/user-chriswood-invasive-group-a-strep.csv",
    system="snomed",
    column="code",
)

strep_a_sore_throat_codes = codelist_from_csv(
    "codelists/user-chriswood-group-a-streptococcal-sore-throat.csv",
    system="snomed",
    column="code",
)
