# ============================================================
# STUDENT PERFORMANCE CLASSIFICATION
# IMPROVED MODEL TRAINING
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "refined_data/Students_Grading_Complete.csv"

FEATURES = [
    "Attendance (%)",
    "Midterm_Score",
    "Assignments_Avg",
    "Quizzes_Avg"
]

TARGET = "Performance"

RANDOM_STATE = 42

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STUDENT PERFORMANCE CLASSIFICATION")
print("IMPROVED MODEL TRAINING")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading refined dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully.")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING REQUIRED COLUMNS")
print("=" * 70)

required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("\nERROR: Missing columns:")
    for col in missing_columns:
        print("-", col)
    raise SystemExit

print("All required columns found.")


# ============================================================
# REMOVE INVALID TARGET ROWS
# ============================================================

df = df.dropna(subset=[TARGET]).copy()

print(f"\nRows available for training: {len(df)}")


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df[FEATURES].copy()
y = df[TARGET].copy()


# ============================================================
# DISPLAY CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PERFORMANCE CLASS DISTRIBUTION")
print("=" * 70)

class_counts = y.value_counts()

print("\nCounts:")
print(class_counts)

print("\nPercentages:")
print((y.value_counts(normalize=True) * 100).round(2))


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# ============================================================
# PREPROCESSING
# ============================================================

# Median imputation + standardization
#
# IMPORTANT:
# Imputer and scaler are fitted ONLY on training data
# through Pipeline, preventing data leakage.

preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]),

    "Decision Tree": Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]),

    "SVM": Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            SVC(
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]),

    "KNN": Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            KNeighborsClassifier()
        )
    ])
}


# ============================================================
# HYPERPARAMETER GRIDS
# ============================================================

param_grids = {

    "Logistic Regression": {
        "model__C": [0.1, 1, 10],
        "model__solver": ["lbfgs"]
    },

    "Decision Tree": {
        "model__max_depth": [3, 5, 7, 10, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 5]
    },

    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10, 15],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
    },

    "SVM": {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["rbf", "linear"],
        "model__gamma": ["scale", "auto"]
    },

    "KNN": {
        "model__n_neighbors": [3, 5, 7, 9, 11],
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan"]
    }
}


# ============================================================
# TRAIN MODELS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING AND TUNING MACHINE LEARNING MODELS")
print("=" * 70)

results = []
trained_models = {}

for name in models:

    print(f"\nTraining: {name}...")

    grid = GridSearchCV(
        estimator=models[name],
        param_grid=param_grids[name],
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=0
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

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

    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"Weighted F1    : {f1_weighted:.4f}")
    print(f"Macro F1       : {f1_macro:.4f}")

    print("Best parameters:")
    print(grid.best_params_)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Weighted_F1": f1_weighted,
        "Macro_F1": f1_macro
    })

    trained_models[name] = best_model

    # Save individual model
    filename = name.lower().replace(" ", "_") + "_improved.pkl"

    joblib.dump(
        best_model,
        os.path.join("models", filename)
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro_F1",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

best_accuracy = results_df.iloc[0]["Accuracy"]
best_macro_f1 = results_df.iloc[0]["Macro_F1"]


print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"\nModel       : {best_model_name}")
print(f"Accuracy    : {best_accuracy:.4f}")
print(f"Accuracy    : {best_accuracy * 100:.2f}%")
print(f"Macro F1    : {best_macro_f1:.4f}")


# ============================================================
# FINAL PREDICTIONS
# ============================================================

y_pred_best = best_model.predict(X_test)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT - BEST MODEL")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred_best,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX - BEST MODEL")
print("=" * 70)

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    y_pred_best,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(cm_df)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

# Feature importance for tree-based models
final_estimator = best_model.named_steps["model"]

if hasattr(final_estimator, "feature_importances_"):

    importances = final_estimator.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print(
        importance_df.to_string(
            index=False,
            formatters={
                "Importance": "{:.4f}".format
            }
        )
    )

# Coefficients for Logistic Regression
elif hasattr(final_estimator, "coef_"):

    coefficients = np.mean(
        np.abs(final_estimator.coef_),
        axis=0
    )

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": coefficients
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print(
        importance_df.to_string(
            index=False,
            formatters={
                "Importance": "{:.4f}".format
            }
        )
    )

else:
    print(
        "\nFeature importance is not directly available "
        "for this model."
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    "results/improved_model_comparison.csv",
    index=False
)

prediction_df = X_test.copy()

prediction_df["Actual_Performance"] = y_test.values
prediction_df["Predicted_Performance"] = y_pred_best

prediction_df.to_csv(
    "results/improved_test_predictions.csv",
    index=False
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/best_model_improved.pkl"
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

with open(
    "models/model_features.txt",
    "w",
    encoding="utf-8"
) as f:

    for feature in FEATURES:
        f.write(feature + "\n")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)

print(f"\nDataset:")
print(f"- {DATA_FILE}")

print(f"\nTotal samples : {len(df)}")
print(f"Training      : {len(X_train)}")
print(f"Testing       : {len(X_test)}")

print("\nFeatures used:")

for i, feature in enumerate(FEATURES, 1):
    print(f"{i}. {feature}")

print("\nTarget:")
print(f"- {TARGET}")

print("\nBest model:")
print(f"- {best_model_name}")

print("\nBest accuracy:")
print(f"- {best_accuracy * 100:.2f}%")

print("\nBest Macro F1:")
print(f"- {best_macro_f1:.4f}")

print("\nSaved files:")
print("- results/improved_model_comparison.csv")
print("- results/improved_test_predictions.csv")
print("- models/best_model_improved.pkl")
print("- models/model_features.txt")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)