# ============================================================
# 05_visualization.py
# STUDENT PERFORMANCE CLASSIFICATION
# DATA & MODEL VISUALIZATION
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "refined_data/Students_Grading_Complete.csv"
MODEL_RESULTS_PATH = "results/improved_model_comparison.csv"
CLASS_RESULTS_PATH = "results/class_wise_performance.csv"
CONFUSION_PATH = "results/confusion_matrix.csv"

OUTPUT_DIR = "results/visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STUDENT PERFORMANCE CLASSIFICATION")
print("DATA & MODEL VISUALIZATION")
print("=" * 70)

# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

try:
    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded successfully.")
    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

except Exception as e:
    print("\nERROR loading dataset:")
    print(e)
    exit()

# ============================================================
# LOAD MODEL RESULTS
# ============================================================

print("\nLoading model comparison results...")

try:
    model_results = pd.read_csv(MODEL_RESULTS_PATH)

    print("Model comparison loaded successfully.")

except Exception as e:
    print("\nERROR loading model comparison:")
    print(e)
    exit()

# ============================================================
# CREATE VISUALIZATION DIRECTORY
# ============================================================

print("\nVisualization output directory:")
print(OUTPUT_DIR)

# ============================================================
# 1. PERFORMANCE CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("1. PERFORMANCE CLASS DISTRIBUTION")
print("=" * 70)

if "Performance" in df.columns:

    performance_counts = df["Performance"].value_counts()

    plt.figure(figsize=(8, 6))

    performance_counts.plot(
        kind="bar"
    )

    plt.title("Student Performance Class Distribution")
    plt.xlabel("Performance Class")
    plt.ylabel("Number of Students")
    plt.xticks(rotation=0)
    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "01_performance_distribution.png"
    )

    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved: {path}")

# ============================================================
# 2. MODEL ACCURACY COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("2. MODEL ACCURACY COMPARISON")
print("=" * 70)

# Find model and accuracy columns safely

model_column = None
accuracy_column = None

for col in model_results.columns:

    if col.lower() == "model":
        model_column = col

    if col.lower() == "accuracy":
        accuracy_column = col

if model_column and accuracy_column:

    plt.figure(figsize=(9, 6))

    plt.bar(
        model_results[model_column],
        model_results[accuracy_column] * 100
    )

    plt.title("Machine Learning Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=20)

    # Add values above bars
    for i, value in enumerate(
        model_results[accuracy_column] * 100
    ):

        plt.text(
            i,
            value + 1,
            f"{value:.2f}%",
            ha="center"
        )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "02_model_accuracy_comparison.png"
    )

    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved: {path}")

# ============================================================
# 3. MODEL METRICS COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("3. MODEL METRICS COMPARISON")
print("=" * 70)

metric_columns = [
    "Precision",
    "Recall",
    "Weighted_F1",
    "Macro_F1"
]

available_metrics = [
    col for col in metric_columns
    if col in model_results.columns
]

if model_column and available_metrics:

    plot_data = model_results.set_index(model_column)[
        available_metrics
    ]

    plt.figure(figsize=(11, 7))

    plot_data.plot(
        kind="bar"
    )

    plt.title("Model Performance Metrics Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.legend(title="Metric")

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "03_model_metrics_comparison.png"
    )

    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved: {path}")

# ============================================================
# 4. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("4. CONFUSION MATRIX")
print("=" * 70)

try:

    cm = pd.read_csv(
        CONFUSION_PATH,
        index_col=0
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=cm.columns,
        yticklabels=cm.index
    )

    plt.title("Confusion Matrix - KNN Model")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "04_confusion_matrix.png"
    )

    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved: {path}")

except Exception as e:

    print("Could not create confusion matrix.")
    print(e)

# ============================================================
# 5. FEATURE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("5. FEATURE DISTRIBUTIONS")
print("=" * 70)

features = [
    "Attendance (%)",
    "Midterm_Score",
    "Assignments_Avg",
    "Quizzes_Avg"
]

for feature in features:

    if feature not in df.columns:
        continue

    plt.figure(figsize=(8, 6))

    plt.hist(
        df[feature].dropna(),
        bins=20
    )

    plt.title(
        f"Distribution of {feature}"
    )

    plt.xlabel(feature)
    plt.ylabel("Number of Students")

    plt.tight_layout()

    safe_name = (
        feature
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "percent")
    )

    path = os.path.join(
        OUTPUT_DIR,
        f"05_distribution_{safe_name}.png"
    )

    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved: {path}")

# ============================================================
# 6. FEATURE VS PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("6. FEATURE VS PERFORMANCE")
print("=" * 70)

if "Performance" in df.columns:

    for feature in features:

        if feature not in df.columns:
            continue

        plt.figure(figsize=(9, 6))

        sns.boxplot(
            data=df,
            x="Performance",
            y=feature
        )

        plt.title(
            f"{feature} vs Student Performance"
        )

        plt.xlabel("Performance")
        plt.ylabel(feature)

        plt.xticks(rotation=10)

        plt.tight_layout()

        safe_name = (
            feature
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("%", "percent")
        )

        path = os.path.join(
            OUTPUT_DIR,
            f"06_{safe_name}_vs_performance.png"
        )

        plt.savefig(path, dpi=300)
        plt.show()
        plt.close()

        print(f"Saved: {path}")

# ============================================================
# 7. FEATURE CORRELATION HEATMAP
# ============================================================

print("\n" + "=" * 70)
print("7. FEATURE CORRELATION HEATMAP")
print("=" * 70)

numeric_features = [
    feature
    for feature in features
    if feature in df.columns
]

if len(numeric_features) > 1:

    correlation = df[numeric_features].corr()

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True
    )

    plt.title(
        "Correlation Between Student Performance Features"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "07_feature_correlation_heatmap.png"
    )

    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved: {path}")

# ============================================================
# 8. AVERAGE FEATURE SCORE BY PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("8. AVERAGE FEATURE SCORE BY PERFORMANCE")
print("=" * 70)

if "Performance" in df.columns:

    available_features = [
        feature
        for feature in features
        if feature in df.columns
    ]

    grouped = df.groupby(
        "Performance"
    )[available_features].mean()

    plt.figure(figsize=(11, 7))

    grouped.plot(
        kind="bar"
    )

    plt.title(
        "Average Academic Indicators by Performance Class"
    )

    plt.xlabel("Performance Class")
    plt.ylabel("Average Score")

    plt.xticks(rotation=0)
    plt.legend(title="Feature")

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "08_average_features_by_performance.png"
    )

    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved: {path}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print("\nAll generated visualizations are stored in:")

print(f"{OUTPUT_DIR}")

print("\nGenerated visualizations include:")

print("1. Performance class distribution")
print("2. Model accuracy comparison")
print("3. Model metrics comparison")
print("4. Confusion matrix")
print("5. Feature distributions")
print("6. Feature vs performance analysis")
print("7. Feature correlation heatmap")
print("8. Average feature score by performance")

print("\nDONE")
print("=" * 70)