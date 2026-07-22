"""
============================================================
Student Academic Performance Prediction System
------------------------------------------------------------
Module : Data Preprocessing
Purpose: Load, inspect, rename, clean and save the dataset
============================================================
"""

import pandas as pd
from pathlib import Path


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(file_path):
    """
    Load the questionnaire dataset from an Excel file.
    """

    try:
        df = pd.read_excel(file_path)

        print("=" * 60)
        print("Dataset loaded successfully.")
        print(f"Rows    : {df.shape[0]}")
        print(f"Columns : {df.shape[1]}")
        print("=" * 60)

        return df

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


# ==========================================================
# INSPECT DATASET
# ==========================================================

def inspect_dataset(df):
    """
    Display basic information about the dataset.
    """

    print("\nDATASET INFORMATION")
    print("=" * 60)

    print(df.info())

    print("\nFirst Five Rows")
    print(df.head())

    print("\nSummary Statistics")
    print(df.describe(include="all"))

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())


# ==========================================================
# RENAME COLUMNS
# ==========================================================

def rename_columns(df):
    """
    Rename columns to shorter names.
    """

    new_columns = {
        "S/N": "S_N",
        "Gender": "Gender",
        "Age": "Age",
        "Level of Study": "Level",
        "Faculty": "Faculty",
        "Department": "Department",
        "Current CGPA": "CGPA",
        "Average CA Score (Previous Semester)": "CA_Score",
        "Average Class Attendance Rate (Previous Semester)": "Attendance",
        "Passed All Courses (Previous Semester)": "Passed_All",
        "Self-Rated Academic Performance": "Academic_Performance",
        "Average Daily Study Hours": "Study_Hours",
        "Average Daily Social Media Hours": "Social_Media_Hours",
        "Average Daily Academic Internet Research Hours": "Internet_Research_Hours",
        "Primary Device Used for Learning": "Learning_Device",
        "Uses Online Educational Resources": "Uses_Online_Resources",
        "Frequency of Using Online Educational Resources": "Resource_Frequency",
        "Social Media Distracts from Studies": "Social_Media_Distraction",
        "Frequency of Completing Assignments Before Deadline": "Assignment_Completion",
        "Participates in Online Academic Discussions/Study Groups": "Online_Discussion",
        "Self-Rated Time Management Skills": "Time_Management",
        "Passed Most Recent Semester (Prediction Class)": "Prediction_Class"
    }

    df = df.rename(columns=new_columns)

    print("\nColumns renamed successfully.")

    return df


# ==========================================================
# CLEAN DATASET
# ==========================================================

def clean_dataset(df):
    """
    Clean the dataset before encoding.
    """

    print("\nCleaning Dataset...")

    # Remove S/N
    df = df.drop(columns=["S_N"])

    # Remove leading/trailing spaces
    object_columns = df.select_dtypes(include="object").columns

    for column in object_columns:
        df[column] = df[column].str.strip()

    # Standardize Faculty names
    faculty = {
        "Management science": "Management Sciences",
        "Basic medical sciences": "Basic Medical Sciences",
        "Computing & Informatics": "Computing and Informatics",
         "Computing and informatics": "Computing and Informatics"

    }

    df["Faculty"] = df["Faculty"].replace(faculty)

    # Standardize Department names
    department = {
        "PHYSIOLOGY": "Physiology",
        "Information systems": "Information System",
        "business management": "Business Administration"
    }

    df["Department"] = df["Department"].replace(department)

    # Standardize Attendance
    df["Attendance"] = df["Attendance"].replace({
        "80%  - 89%": "80% - 89%"
    })

    # Fill missing numeric values with median
    numeric_columns = df.select_dtypes(include="number").columns

    df[numeric_columns] = df[numeric_columns].fillna(
        df[numeric_columns].median()
    )

    print("Cleaning completed.")

    return df


# ==========================================================
# VALIDATE DATASET
# ==========================================================

def validate_dataset(df):
    """
    Check the cleaned dataset.
    """

    print("\nValidation")
    print("=" * 60)

    print("Missing Values")
    print(df.isnull().sum())

    print("\nDataset Shape")
    print(df.shape)


# ==========================================================
# SAVE DATASET
# ==========================================================

def save_dataset(df):
    """
    Save the cleaned dataset.
    """

    output_folder = Path("data/processed")

    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / "cleaned_dataset.xlsx"

    df.to_excel(output_file, index=False)

    print(f"\nDataset saved successfully to:\n{output_file}")
