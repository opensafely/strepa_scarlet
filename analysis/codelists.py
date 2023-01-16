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
