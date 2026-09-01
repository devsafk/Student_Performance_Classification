import pandas as pd
import numpy as np
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# STEP 1: LOAD FROZEN DATA
# ============================================================

print("=" * 60)
print("STUDENT PERFORMANCE CLASSIFICATION - MODEL TRAINING")
print("=" * 60)

X_train = pd.read_csv("frozen_data/X_train_FROZEN.csv")
X_test = pd.read_csv("frozen_data/X_test_FROZEN.csv")

y_train = pd.read_csv("frozen_data/y_train_FROZEN.csv").squeeze()
y_test = pd.read_csv("frozen_data/y_test_FROZEN.csv").squeeze()

print("\nFrozen data loaded successfully.")

print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)


# ============================================================
# STEP 2: DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Support Vector Machine": SVC(
        kernel="rbf",
        random_state=42
    ),

    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5
    )
}


# ============================================================
# STEP 3: CROSS-VALIDATION ON TRAINING DATA
# ============================================================

print("\n" + "=" * 60)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 60)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = {}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    cv_results[name] = scores.mean()

    print(
        f"{name:<25} "
        f"Mean F1: {scores.mean():.4f} "
        f"(± {scores.std():.4f})"
    )


# ============================================================
# STEP 4: SELECT BEST MODEL
# ============================================================

best_model_name = max(
    cv_results,
    key=cv_results.get
)

best_cv_score = cv_results[best_model_name]

print("\n" + "=" * 60)
print("BEST MODEL FROM CROSS-VALIDATION")
print("=" * 60)

print("Model:", best_model_name)
print(f"Mean CV Macro F1: {best_cv_score:.4f}")


# ============================================================
# STEP 5: TRAIN BEST MODEL ON COMPLETE TRAINING DATA
# ============================================================

best_model = models[best_model_name]

best_model.fit(
    X_train,
    y_train
)

print("\nBest model trained on all 4,000 training samples.")


# ============================================================
# STEP 6: FINAL EVALUATION ON FROZEN TEST DATA
# ============================================================

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="macro"
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)


print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy        : {accuracy:.4f}")
print(f"Macro Precision : {precision:.4f}")
print(f"Macro Recall    : {recall:.4f}")
print(f"Macro F1-Score  : {f1:.4f}")


# ============================================================
# STEP 7: CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# STEP 8: CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

labels = [
    "Needs Improvement",
    "Average Performer",
    "High Performer"
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(cm_df)


# ============================================================
# STEP 9: SAVE MODEL
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

joblib.dump(
    best_model,
    "models/best_model_FROZEN.pkl"
)


# ============================================================
# STEP 10: SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": list(cv_results.keys()),
    "Cross_Validation_Macro_F1": list(cv_results.values())
})

results = results.sort_values(
    "Cross_Validation_Macro_F1",
    ascending=False
)

results.to_csv(
    "results/model_comparison.csv",
    index=False
)

pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1"
        ],
        "Score": [
            accuracy,
            precision,
            recall,
            f1
        ]
    }
).to_csv(
    "results/best_model_metrics.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)

print("\nSaved:")
print("models/best_model_FROZEN.pkl")
print("results/model_comparison.csv")
print("results/best_model_metrics.csv")