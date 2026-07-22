from src.model import (
    load_encoded_dataset,
    split_dataset,
    train_model,
    evaluate_model,
    plot_confusion_matrix_chart,
    plot_feature_importance,
    visualize_tree,
    save_model
)


def main():

    print("=" * 70)
    print("DECISION TREE MODEL")
    print("=" * 70)

    # STEP 1
    print("\nSTEP 1: Loading Encoded Dataset...")
    df = load_encoded_dataset()

    # STEP 2
    print("\nSTEP 2: Splitting Dataset...")
    X_train, X_test, y_train, y_test = split_dataset(df)

    # STEP 3
    print("\nSTEP 3: Training Model...")
    model = train_model(X_train, y_train)

    # STEP 4
    print("\nSTEP 4: Evaluating Model...")
    evaluate_model(
        model,
        X_test,
        y_test
    )

    # STEP 5
    print("\nSTEP 5: Confusion Matrix...")
    plot_confusion_matrix_chart(
        model,
        X_test,
        y_test
    )

    # STEP 6
    print("\nSTEP 6: Feature Importance...")
    plot_feature_importance(
        model,
        X_train
    )

    # STEP 7
    print("\nSTEP 7: Decision Tree Visualization...")
    visualize_tree(
        model,
        X_train
    )

    # STEP 8
    print("\nSTEP 8: Saving Model...")
    save_model(model)

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
    