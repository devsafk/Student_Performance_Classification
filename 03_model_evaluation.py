# ============================================================
# STUDENT PERFORMANCE CLASSIFICATION
# MODEL EVALUATION
# ============================================================

import os
import warnings
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "refined_data/Students_Grading_Complete.csv"
MODEL_PATH = "models/best_model_improved.pkl"
FEATURE_FILE = "models/model_features.txt"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("STUDENT PERFORMANCE CLASSIFICATION")
print("MODEL EVALUATION")
print("=" * 75)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading refined dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Rows    :", len(df))
print("Columns :", len(df))


# ============================================================
# REQUIRED FEATURES
# ============================================================

features = [
    "Attendance (%)",
    "Midterm_Score",
    "Assignments_Avg",
    "Quizzes_Avg"
]

target = "Performance"


print("\n" + "=" * 75)
print("FEATURES AND TARGET")
print("=" * 75)

print("\nFeatures used:")

for feature in features:
    print("-", feature)

print("\nTarget:")
print("-", target)


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = features + [target]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("\nERROR: Missing columns:")
    for col in missing_columns:
        print("-", col)
    raise SystemExit


print("\nAll required columns found.")


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 75)
print("DATA CLEANING")
print("=" * 75)

print("\nMissing values before cleaning:")

print(df[features].isnull().sum())

for feature in features:
    if df[feature].isnull().sum() > 0:
        df[feature] = df[feature].fillna(
            df[feature].median()
        )

df = df.dropna(subset=[target])

print("\nMissing values after cleaning:")
print(df[features].isnull().sum())


# ============================================================
# PREPARE X AND Y
# ============================================================

X = df[features]
y = df[target]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 75)
print("TRAIN / TEST SPLIT")
print("=" * 75)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 75)
print("LOADING BEST MODEL")
print("=" * 75)

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model file not found:")
    print(MODEL_PATH)
    raise SystemExit

model = joblib.load(MODEL_PATH)

print("\nBest model loaded successfully.")
print("Model:", type(model).__name__)


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 75)
print("GENERATING PREDICTIONS")
print("=" * 75)

y_pred = model.predict(X_test)

print("\nPredictions generated successfully.")


# ============================================================
# PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# DISPLAY METRICS
# ============================================================

print("\n" + "=" * 75)
print("MODEL PERFORMANCE")
print("=" * 75)

print(f"\nAccuracy       : {accuracy:.4f}")
print(f"Accuracy       : {accuracy * 100:.2f}%")

print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"Weighted F1    : {weighted_f1:.4f}")
print(f"Macro F1       : {macro_f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print("\n" + report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

classes = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)

print("\n")
print(cm_df)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.csv"
)

cm_df.to_csv(cm_path)

print("\nConfusion matrix saved to:")
print(cm_path)


# ============================================================
# SAVE EVALUATION METRICS
# ============================================================

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "Weighted F1",
        "Macro F1"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        weighted_f1,
        macro_f1
    ]
})

metrics_path = os.path.join(
    RESULTS_DIR,
    "evaluation_metrics.csv"
)

metrics.to_csv(metrics_path, index=False)

print("\nEvaluation metrics saved to:")
print(metrics_path)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_df = X_test.copy()

prediction_df["Actual_Performance"] = y_test.values
prediction_df["Predicted_Performance"] = y_pred

prediction_df["Correct"] = (
    prediction_df["Actual_Performance"]
    == prediction_df["Predicted_Performance"]
)

prediction_path = os.path.join(
    RESULTS_DIR,
    "evaluation_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)

print("\nEvaluation predictions saved to:")
print(prediction_path)


# ============================================================
# CORRECT / INCORRECT PREDICTIONS
# ============================================================

correct = prediction_df["Correct"].sum()
incorrect = len(prediction_df) - correct

print("\n" + "=" * 75)
print("PREDICTION SUMMARY")
print("=" * 75)

print("\nCorrect predictions  :", correct)
print("Incorrect predictions:", incorrect)
print("Total predictions    :", len(prediction_df))


# ============================================================
# CLASS-WISE PERFORMANCE
# ============================================================

print("\n" + "=" * 75)
print("CLASS-WISE PERFORMANCE")
print("=" * 75)

class_report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

class_rows = []

for class_name in classes:
    class_rows.append({
        "Performance": class_name,
        "Precision": class_report[class_name]["precision"],
        "Recall": class_report[class_name]["recall"],
        "F1_Score": class_report[class_name]["f1-score"],
        "Support": int(class_report[class_name]["support"])
    })

class_df = pd.DataFrame(class_rows)

print("\n")
print(class_df.to_string(index=False))


# ============================================================
# SAVE CLASS-WISE PERFORMANCE
# ============================================================

class_path = os.path.join(
    RESULTS_DIR,
    "class_wise_performance.csv"
)

class_df.to_csv(
    class_path,
    index=False
)

print("\nClass-wise performance saved to:")
print(class_path)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("MODEL EVALUATION COMPLETE")
print("=" * 75)

print("\nModel              :", type(model).__name__)
print("Dataset            :", DATA_PATH)
print("Total samples      :", len(df))
print("Training samples   :", len(X_train))
print("Testing samples    :", len(X_test))

print("\nFeatures:")
for i, feature in enumerate(features, 1):
    print(f"{i}. {feature}")

print("\nTarget:")
print("-", target)

print("\nFinal Results:")
print(f"- Accuracy       : {accuracy * 100:.2f}%")
print(f"- Precision      : {precision:.4f}")
print(f"- Recall         : {recall:.4f}")
print(f"- Weighted F1    : {weighted_f1:.4f}")
print(f"- Macro F1       : {macro_f1:.4f}")

print("\nOutput files:")
print("-", cm_path)
print("-", metrics_path)
print("-", prediction_path)
print("-", class_path)

print("\n" + "=" * 75)
print("DONE")
print("=" * 75)