from src.eda import (
    load_cleaned_dataset,
    dataset_overview,
    descriptive_statistics,
    frequency_tables,
    plot_gender_distribution,
    plot_level_distribution,
    plot_faculty_distribution,
    plot_cgpa_distribution,
    plot_attendance_distribution,
    plot_academic_performance_distribution,
    plot_prediction_class_distribution,
    plot_age_distribution,
    plot_ca_score_distribution,
    plot_study_hours_distribution,
    plot_social_media_distribution,
    plot_research_hours_distribution
)


def main():

    print("=" * 70)
    print("STUDENT ACADEMIC PERFORMANCE PREDICTION SYSTEM")
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    # ============================================
    # STEP 1: Load Cleaned Dataset
    # ============================================
    print("\nSTEP 1: LOADING CLEANED DATASET...")

    df = load_cleaned_dataset()

    # ============================================
    # STEP 2: Dataset Overview
    # ============================================
    print("\nSTEP 2: DATASET OVERVIEW...")

    dataset_overview(df)

    # ============================================
    # STEP 3: Descriptive Statistics
    # ============================================
    print("\nSTEP 3: DESCRIPTIVE STATISTICS...")

    descriptive_statistics(df)

    # ============================================
    # STEP 4: Frequency Tables
    # ============================================
    print("\nSTEP 4: FREQUENCY TABLES...")

    frequency_tables(df)

    # ============================================
    # STEP 5: Data Visualization
    # ============================================
    print("\nSTEP 5: GENERATING VISUALIZATIONS...")

    plot_gender_distribution(df)

    plot_level_distribution(df)

    plot_faculty_distribution(df)

    plot_cgpa_distribution(df)

    plot_attendance_distribution(df)

    plot_academic_performance_distribution(df)

    plot_prediction_class_distribution(df)

# ============================================
# STEP 6: Numerical Visualizations
# ============================================

    print("\nSTEP 6: NUMERICAL VISUALIZATIONS...")

    plot_age_distribution(df)

    plot_ca_score_distribution(df)

    plot_study_hours_distribution(df)

    plot_social_media_distribution(df)

    plot_research_hours_distribution(df)

    print("\n" + "=" * 70)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()