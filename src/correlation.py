"""
============================================================
Student Academic Performance Prediction System
------------------------------------------------------------
Module: Correlation Analysis
Purpose: Analyze relationships among variables
============================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_encoded_dataset():
    """
    Load encoded dataset.
    """

    file_path = "data/processed/encoded_dataset.xlsx"

    df = pd.read_excel(file_path)

    print("=" * 60)
    print("ENCODED DATASET LOADED")
    print("=" * 60)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df

def correlation_matrix(df):
    """
    Compute correlation matrix.
    """

    print("\nGenerating Correlation Matrix...")

    correlation = df.corr(numeric_only=True)

    print(correlation.round(3))

    return correlation

def save_correlation_matrix(correlation):

    output = Path("outputs")

    output.mkdir(exist_ok=True)

    correlation.round(3).to_excel(

        output / "correlation_matrix.xlsx"

    )

    print("\nCorrelation matrix saved.")

def plot_heatmap(correlation):
    """
    Plot correlation heatmap.
    """

    plt.figure(figsize=(14,10))

    plt.imshow(
        correlation,
        cmap="coolwarm",
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    output = Path("outputs/figures")

    output.mkdir(parents=True, exist_ok=True)

    plt.savefig(

        output / "correlation_heatmap.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.show()

    print("\nHeatmap saved successfully.")

def target_correlation(df):
    """
    Display correlation with target variable.
    """

    print("\nCorrelation with Prediction Class")

    corr = df.corr(numeric_only=True)["Prediction_Class"]

    corr = corr.sort_values(ascending=False)

    print(corr)

    corr.to_excel(

        "outputs/target_correlation.xlsx"

    )

