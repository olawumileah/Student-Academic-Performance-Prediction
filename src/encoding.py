"""
============================================================
Student Academic Performance Prediction System
------------------------------------------------------------
Module: Data Encoding
Purpose: Convert categorical variables into numerical values
============================================================
"""

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import joblib

def load_cleaned_dataset():
    """
    Load the cleaned dataset.
    """

    file_path = "data/processed/cleaned_dataset.xlsx"

    df = pd.read_excel(file_path)

    print("=" * 60)
    print("CLEANED DATASET LOADED")
    print("=" * 60)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df

def encode_dataset(df):
    """
    Encode categorical variables.
    """

    print("\nEncoding dataset...")

    # Remove leading/trailing spaces from all text columns
    object_columns = df.select_dtypes(include="object").columns

    for column in object_columns:
        df[column] = df[column].astype(str).str.strip()

    mappings = {

        "Gender": {
            "Female": 0,
            "Male": 1
        },

        "Level": {
            "100 Level": 1,
            "200 Level": 2,
            "300 Level": 3,
            "400 Level": 4,
            "500 Level": 5
        },

        "CGPA": {
            "Below 1.50": 1,
            "1.50 - 2.49": 2,
            "2.50 - 3.49": 3,
            "3.50 - 4.49": 4,
            "4.50 - 5.00": 5
        },

        "Attendance": {
            "Below 50%": 1,
            "50% - 59%": 2,
            "60% - 69%": 3,
            "70% - 79%": 4,
            "80% - 89%": 5,
            "90% - 100%": 6
        },

        "Passed_All": {
            "No": 0,
            "Yes": 1
        },

        "Academic_Performance": {
            "Poor": 1,
            "Fair": 2,
            "Good": 3,
            "Very Good": 4,
            "Excellent": 5
        },

       "Learning_Device": {
            "Smartphone": 0,
            "Laptop": 1,
            "Desktop Computer": 2,
            "Tablet": 3
        },
        "Uses_Online_Resources": {
            "No": 0,
            "Yes": 1
        },

        "Resource_Frequency": {
            "Never": 1,
            "Rarely": 2,
            "Sometimes": 3,
            "Often": 4,
            "Very Often": 5,
            "Always": 6
        },

        "Social_Media_Distraction": {
            "Strongly Disagree": 1,
            "Disagree": 2,
            "Neutral": 3,
            "Agree": 4,
            "Strongly Agree": 5
        },

        "Assignment_Completion": {
            "Never": 1,
            "Rarely": 2,
            "Sometimes": 3,
            "Often": 4,
            "Always": 5
        },

        "Online_Discussion": {
            "No": 0,
            "Yes": 1
        },

        "Time_Management": {
            "Poor": 1,
            "Fair": 2,
            "Good": 3,
            "Very Good": 4,
            "Excellent": 5
        },

        "Prediction_Class": {
            "No": 0,
            "Yes": 1
        }

    }

    # Encode each column and report any unmapped values
    for column, mapping in mappings.items():

        unique_values = set(df[column].dropna().unique())

        unmapped = unique_values - set(mapping.keys())

        if len(unmapped) > 0:
            print(f"\nWARNING: Unmapped values found in '{column}'")
            print(unmapped)

        df[column] = df[column].map(mapping)
        # =====================================================
    # LABEL ENCODE FACULTY
    # =====================================================

    faculty_encoder = LabelEncoder()

    df["Faculty"] = faculty_encoder.fit_transform(df["Faculty"])

    # =====================================================
    # LABEL ENCODE DEPARTMENT
    # =====================================================

    department_encoder = LabelEncoder()

    df["Department"] = department_encoder.fit_transform(df["Department"])

    # =====================================================
    # SAVE ENCODERS
    # =====================================================

    Path("models").mkdir(exist_ok=True)

    joblib.dump(faculty_encoder, "models/faculty_encoder.pkl")
    joblib.dump(department_encoder, "models/department_encoder.pkl")

    print("\nFaculty and Department encoded successfully.")


    print("\nEncoding completed successfully.")

    return df

def validate_encoding(df):
    """
    Validate encoded dataset.
    """

    print("\n" + "=" * 60)
    print("ENCODED DATASET INFORMATION")
    print("=" * 60)

    print(df.head())

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    # Show rows containing missing values after encoding
    if df.isnull().sum().sum() > 0:

        print("\nRows containing missing values after encoding:")

        print(df[df.isnull().any(axis=1)])

def save_encoded_dataset(df):
    """
    Save encoded dataset.
    """

    output = Path("data/processed")

    output.mkdir(parents=True, exist_ok=True)

    df.to_excel(

        output / "encoded_dataset.xlsx",

        index=False

    )

    print("\nEncoded dataset saved successfully.")
