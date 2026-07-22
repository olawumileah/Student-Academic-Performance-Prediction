"""
============================================================
Student Academic Performance Prediction System
------------------------------------------------------------
Module: Decision Tree Model
Purpose: Train and Evaluate Academic Performance Prediction
============================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

def load_encoded_dataset():
    """
    Load the encoded dataset.
    """

    file_path = "data/processed/encoded_dataset.xlsx"

    df = pd.read_excel(file_path)

    print("=" * 60)
    print("ENCODED DATASET LOADED")
    print("=" * 60)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df

def split_dataset(df):
    """
    Split dataset into training and testing sets.
    """

    print("\nSplitting Dataset...")

    X = df.drop(columns=["Prediction_Class"])

    y = df["Prediction_Class"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    print(f"Training Samples : {len(X_train)}")

    print(f"Testing Samples  : {len(X_test)}")

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train Decision Tree Classifier.
    """

    print("\nTraining Decision Tree Model...")

    model = DecisionTreeClassifier(

        criterion="gini",

        max_depth=5,

        random_state=42

    )

    model.fit(

        X_train,

        y_train

    )

    print("Model trained successfully.")

    return model

def save_model(model):
    """
    Save trained model.
    """

    output = Path("models")

    output.mkdir(exist_ok=True)

    joblib.dump(

        model,

        output / "decision_tree_model.pkl"

    )

    print("\nDecision Tree model saved successfully.")

def evaluate_model(model, X_test, y_test):
    """
    Evaluate Decision Tree model.
    """

    print("\nEvaluating Model...")

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report")

    print(classification_report(y_test, y_pred))

    return y_pred

def plot_confusion_matrix_chart(model, X_test, y_test):
    """
    Plot confusion matrix.
    """

    print("\nGenerating Confusion Matrix...")

    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=["Fail", "Pass"]

    )

    plt.figure(figsize=(6,6))

    display.plot(cmap="Blues")

    output = Path("outputs/figures")

    output.mkdir(parents=True, exist_ok=True)

    plt.savefig(

        output / "confusion_matrix.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.show()

    print("Confusion Matrix saved successfully.")

def plot_feature_importance(model, X_train):
    """
    Plot feature importance.
    """

    print("\nGenerating Feature Importance...")

    importance = pd.Series(

        model.feature_importances_,

        index=X_train.columns

    )

    importance = importance.sort_values()

    plt.figure(figsize=(10,8))

    importance.plot(kind="barh")

    plt.title("Feature Importance")

    plt.tight_layout()

    output = Path("outputs/figures")

    output.mkdir(parents=True, exist_ok=True)

    plt.savefig(

        output / "feature_importance.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.show()

    print("Feature Importance saved successfully.")

def visualize_tree(model, X_train):
    """
    Plot Decision Tree.
    """

    print("\nGenerating Decision Tree...")

    plt.figure(figsize=(24,12))

    plot_tree(

        model,

        feature_names=X_train.columns,

        class_names=["Fail", "Pass"],

        filled=True,

        rounded=True,

        fontsize=8

    )

    output = Path("outputs/figures")

    output.mkdir(parents=True, exist_ok=True)

    plt.savefig(

        output / "decision_tree.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.show()

    print("Decision Tree saved successfully.")