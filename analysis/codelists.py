from cohortextractor import (
    codelist_from_csv,
)

ethnicity_codes = codelist_from_csv(
    "codelists/opensafely-ethnicity-snomed-0removed.csv",
    system="snomed",
    column="snomedcode",
    category_column="Grouping_6",
)

dmard_codes = codelist_from_csv(
    "codelists/opensafely-dmards.csv",
    system="snomed",
    column="snomed_id",
)

allmed_review_codes = codelist_from_csv(
    "codelists/user-chriswood-all-medication-reviews.csv",
    system="snomed",
    column="code",
)