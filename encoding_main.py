from src.encoding import (
    load_cleaned_dataset,
    encode_dataset,
    validate_encoding,
    save_encoded_dataset
)


def main():

    print("=" * 70)
    print("STUDENT ACADEMIC PERFORMANCE PREDICTION SYSTEM")
    print("DATA ENCODING")
    print("=" * 70)

    print("\nSTEP 1: LOADING CLEANED DATASET...")

    df = load_cleaned_dataset()

    print("\nSTEP 2: ENCODING DATASET...")

    df = encode_dataset(df)

    print("\nSTEP 3: VALIDATING DATASET...")

    validate_encoding(df)

    print("\nSTEP 4: SAVING ENCODED DATASET...")

    save_encoded_dataset(df)

    print("\n" + "=" * 70)
    print("ENCODING COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()