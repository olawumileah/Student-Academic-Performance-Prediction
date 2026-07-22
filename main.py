from src.preprocessing import (
    load_dataset,
    inspect_dataset,
    rename_columns,
    clean_dataset,
    validate_dataset,
    save_dataset
)


def main():

    print("=" * 70)
    print("STUDENT ACADEMIC PERFORMANCE PREDICTION SYSTEM")
    print("=" * 70)

    dataset_path = "data/raw/lautech_survey_data.xlsx"

    # ======================================================
    # STEP 1 - LOAD DATASET
    # ======================================================

    print("\nSTEP 1: LOADING DATASET...")

    df = load_dataset(dataset_path)

    if df is None:
        print("\nDataset could not be loaded.")
        return

    # ======================================================
    # STEP 2 - INSPECT DATASET
    # ======================================================

    print("\nSTEP 2: INSPECTING DATASET...")

    inspect_dataset(df)

    # ======================================================
    # STEP 3 - RENAME COLUMNS
    # ======================================================

    print("\nSTEP 3: RENAMING COLUMNS...")

    df = rename_columns(df)

    # ======================================================
    # STEP 4 - CLEAN DATASET
    # ======================================================

    print("\nSTEP 4: CLEANING DATASET...")

    df = clean_dataset(df)

    # ======================================================
    # STEP 5 - VALIDATE DATASET
    # ======================================================

    print("\nSTEP 5: VALIDATING DATASET...")

    validate_dataset(df)

    # ======================================================
    # STEP 6 - SAVE DATASET
    # ======================================================

    print("\nSTEP 6: SAVING CLEANED DATASET...")

    save_dataset(df)

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()