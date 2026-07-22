from src.correlation import (
    load_encoded_dataset,
    correlation_matrix,
    save_correlation_matrix,
    plot_heatmap,
    target_correlation
)


def main():

    print("="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)

    df = load_encoded_dataset()

    correlation = correlation_matrix(df)

    save_correlation_matrix(correlation)

    plot_heatmap(correlation)

    target_correlation(df)

    print("\nCorrelation Analysis Completed Successfully.")


if __name__ == "__main__":
    main()