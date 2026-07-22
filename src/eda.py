"""
============================================================
Student Academic Performance Prediction System
------------------------------------------------------------
Module : Exploratory Data Analysis (EDA)
============================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FIGURE_FOLDER = Path("outputs/figures")
FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)


def load_cleaned_dataset():

    dataset_path = Path("data/processed/cleaned_dataset.xlsx")

    df = pd.read_excel(dataset_path)

    print("=" * 60)
    print("CLEANED DATASET LOADED")
    print("=" * 60)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df

def dataset_overview(df):

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print("\nDataset Shape")
    print(df.shape)

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())


def descriptive_statistics(df):
    """
    Display and save descriptive statistics.
    """

    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)

    numerical_columns = [
        "Age",
        "CA_Score",
        "Study_Hours",
        "Social_Media_Hours",
        "Internet_Research_Hours"
    ]

    statistics = df[numerical_columns].describe().T

    statistics = statistics[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ]

    statistics = statistics.round(2)

    print(statistics)

    # ==========================
    # Save table
    # ==========================

    output_folder = Path("outputs/tables")
    output_folder.mkdir(parents=True, exist_ok=True)

    statistics.to_excel(
        output_folder / "descriptive_statistics.xlsx"
    )

    print("\nDescriptive statistics saved successfully.")

    return statistics


def frequency_tables(df):
    """
    Generate and save frequency tables for all categorical variables.
    """

    print("\n" + "=" * 60)
    print("FREQUENCY DISTRIBUTION TABLES")
    print("=" * 60)

    categorical_columns = [
        "Gender",
        "Level",
        "Faculty",
        "Department",
        "CGPA",
        "Attendance",
        "Passed_All",
        "Academic_Performance",
        "Learning_Device",
        "Uses_Online_Resources",
        "Resource_Frequency",
        "Social_Media_Distraction",
        "Assignment_Completion",
        "Online_Discussion",
        "Time_Management",
        "Prediction_Class"
    ]

    output_folder = Path("outputs/tables")
    output_folder.mkdir(parents=True, exist_ok=True)

    for column in categorical_columns:

        frequency = df[column].value_counts().sort_index()

        percentage = round(
            (frequency / len(df)) * 100,
            2
        )

        table = frequency.to_frame(name="Frequency")

        table["Percentage (%)"] = percentage

        print("\n" + "-" * 60)
        print(column)
        print(table)

        filename = column.lower().replace(" ", "_") + "_frequency.xlsx"

        table.to_excel(output_folder / filename)

    print("\nAll frequency tables saved successfully.")

def plot_gender_distribution(df):
    """
    Plot and save gender distribution.
    """

    print("\nGenerating Gender Distribution Chart...")

    counts = df["Gender"].value_counts()

    plt.figure(figsize=(8,5))

    counts.plot(kind="bar")

    plt.title("Gender Distribution")

    plt.xlabel("Gender")

    plt.ylabel("Number of Students")

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        FIGURE_FOLDER / "gender_distribution.png",
        dpi=300
    )

    plt.show()

    print("Gender distribution saved successfully.")

def plot_level_distribution(df):
    """
    Plot and save Level distribution.
    """

    print("\nGenerating Level Distribution Chart...")

    counts = df["Level"].value_counts()

    plt.figure(figsize=(8,5))

    counts.plot(kind="bar")

    plt.title("Level Distribution")

    plt.xlabel("Level")

    plt.ylabel("Number of Students")

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        FIGURE_FOLDER / "level_distribution.png",
        dpi=300
    )

    plt.show()

    print("Level distribution saved successfully.")

def plot_faculty_distribution(df):
    """
    Plot and save Faculty distribution.
    """

    print("\nGenerating Faculty Distribution Chart...")

    counts = df["Faculty"].value_counts()

    plt.figure(figsize=(12,6))

    counts.plot(kind="bar")

    plt.title("Faculty Distribution")

    plt.xlabel("Faculty")

    plt.ylabel("Number of Students")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        FIGURE_FOLDER / "faculty_distribution.png",
        dpi=300
    )

    plt.show()

    print("Faculty distribution saved successfully.")

def plot_cgpa_distribution(df):
    """
    Plot and save CGPA distribution.
    """

    print("\nGenerating CGPA Distribution Chart...")

    counts = df["CGPA"].value_counts()

    plt.figure(figsize=(8,5))

    counts.plot(kind="bar")

    plt.title("CGPA Distribution")

    plt.xlabel("CGPA")

    plt.ylabel("Number of Students")

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        FIGURE_FOLDER / "cgpa_distribution.png",
        dpi=300
    )

    plt.show()

    print("CGPA distribution saved successfully.")

def plot_attendance_distribution(df):
    """
    Plot and save Attendance distribution.
    """

    print("\nGenerating Attendance Distribution Chart...")

    counts = df["Attendance"].value_counts()

    plt.figure(figsize=(8,5))

    counts.plot(kind="bar")

    plt.title("Attendance Distribution")

    plt.xlabel("Attendance")

    plt.ylabel("Number of Students")

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        FIGURE_FOLDER / "attendance_distribution.png",
        dpi=300
    )

    plt.show()

    print("Attendance distribution saved successfully.")

def plot_academic_performance_distribution(df):
    """
    Plot and save Academic Performance distribution.
    """

    print("\nGenerating Academic Performance Distribution Chart...")

    counts = df["Academic_Performance"].value_counts()

    plt.figure(figsize=(8,5))

    counts.plot(kind="bar")

    plt.title("Academic Performance Distribution")

    plt.xlabel("Academic Performance")

    plt.ylabel("Number of Students")

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        FIGURE_FOLDER / "academic_performance_distribution.png",
        dpi=300
    )

    plt.show()

    print("Academic Performance distribution saved successfully.")

def plot_prediction_class_distribution(df):
    """
    Plot and save Prediction Class distribution.
    """

    print("\nGenerating Prediction Class Distribution Chart...")

    counts = df["Prediction_Class"].value_counts()

    plt.figure(figsize=(8,5))

    counts.plot(kind="bar")

    plt.title("Prediction Class Distribution")

    plt.xlabel("Prediction Class")

    plt.ylabel("Number of Students")

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        FIGURE_FOLDER / "prediction_class_distribution.png",
        dpi=300
    )

    plt.show()

    print("Prediction Class distribution saved successfully.")

def plot_age_distribution(df):
    """
    Plot and save Age distribution.
    """

    print("\nGenerating Age Distribution Chart...")

    plt.figure(figsize=(8,5))

    plt.hist(df["Age"], bins=8)

    plt.title("Age Distribution")

    plt.xlabel("Age")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        FIGURE_FOLDER / "age_distribution.png",
        dpi=300
    )

    plt.show()

    print("Age distribution saved successfully.")

def plot_ca_score_distribution(df):
    """
    Plot and save CA Score distribution.
    """

    print("\nGenerating CA Score Distribution Chart...")

    plt.figure(figsize=(8,5))

    plt.hist(df["CA_Score"], bins=10)

    plt.title("CA Score Distribution")

    plt.xlabel("CA Score")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        FIGURE_FOLDER / "ca_score_distribution.png",
        dpi=300
    )

    plt.show()

    print("CA Score distribution saved successfully.")

def plot_study_hours_distribution(df):
    """
    Plot and save Study Hours distribution.
    """

    print("\nGenerating Study Hours Distribution Chart...")

    plt.figure(figsize=(8,5))

    plt.hist(df["Study_Hours"], bins=10)

    plt.title("Study Hours Distribution")

    plt.xlabel("Study Hours")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        FIGURE_FOLDER / "study_hours_distribution.png",
        dpi=300
    )

    plt.show()

    print("Study Hours distribution saved successfully.")

def plot_social_media_distribution(df):
    """
    Plot and save Social Media Hours distribution.
    """

    print("\nGenerating Social Media Hours Distribution Chart...")

    plt.figure(figsize=(8,5))

    plt.hist(df["Social_Media_Hours"], bins=10)

    plt.title("Social Media Hours Distribution")

    plt.xlabel("Social Media Hours")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        FIGURE_FOLDER / "social_media_distribution.png",
        dpi=300
    )

    plt.show()

    print("Social Media distribution saved successfully.")

def plot_research_hours_distribution(df):
    """
    Plot and save Internet Research Hours distribution.
    """

    print("\nGenerating Internet Research Hours Distribution Chart...")

    plt.figure(figsize=(8,5))

    plt.hist(df["Internet_Research_Hours"], bins=10)

    plt.title("Internet Research Hours Distribution")

    plt.xlabel("Research Hours")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        FIGURE_FOLDER / "research_hours_distribution.png",
        dpi=300
    )

    plt.show()

    print("Research Hours distribution saved successfully.")

