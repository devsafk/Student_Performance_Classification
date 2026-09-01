import pandas as pd
import numpy as np

# ============================================================
# STEP 1: LOAD THE COMPLETE DATASET
# ============================================================

files = [
    "Students_Grading_Dataset1.csv",
    "Students_Grading_Dataset2.csv",
    "Students_Grading_Dataset3.csv",
    "Students_Grading_Dataset4.csv",
    "Students_Grading_Dataset5.csv"
]

dfs = []

for file in files:
    print(f"Loading: {file}")
    temp = pd.read_csv(file)
    dfs.append(temp)

# Combine all five parts
df = pd.concat(dfs, ignore_index=True)

print("\n" + "=" * 50)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 50)

print("Total rows:", len(df))
print("Total columns:", len(df.columns))

print("\nColumn names:")
for i, column in enumerate(df.columns, 1):
    print(f"{i}. {column}")
# ============================================================
# STEP 2: DATA QUALITY AUDIT
# ============================================================

print("\n" + "=" * 50)
print("DATA QUALITY AUDIT")
print("=" * 50)

# ------------------------------------------------------------
# 2.1 DATA TYPES
# ------------------------------------------------------------

print("\n--- DATA TYPES ---")
print(df.dtypes)


# ------------------------------------------------------------
# 2.2 MISSING VALUES
# ------------------------------------------------------------

print("\n--- MISSING VALUES ---")

missing = df.isnull().sum()

missing_percentage = (missing / len(df)) * 100

missing_report = pd.DataFrame({
    "Missing_Count": missing,
    "Missing_Percentage": missing_percentage
})

print(missing_report[missing_report["Missing_Count"] > 0])


# ------------------------------------------------------------
# 2.3 DUPLICATE RECORDS
# ------------------------------------------------------------

print("\n--- DUPLICATES ---")

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)


# ------------------------------------------------------------
# 2.4 NUMERICAL SUMMARY
# ------------------------------------------------------------

print("\n--- NUMERICAL SUMMARY ---")

print(df.describe())


# ------------------------------------------------------------
# 2.5 CHECK OUR FOUR SELECTED FEATURES
# ------------------------------------------------------------

features = [
    "Attendance (%)",
    "Midterm_Score",
    "Assignments_Avg",
    "Quizzes_Avg"
]

print("\n--- SELECTED FEATURES ---")

print(df[features].describe())


# ------------------------------------------------------------
# 2.6 CHECK FINAL SCORE
# ------------------------------------------------------------

print("\n--- FINAL SCORE ---")

print(df["Final_Score"].describe())


# ------------------------------------------------------------
# 2.7 CHECK INVALID VALUES
# ------------------------------------------------------------

print("\n--- INVALID VALUE CHECK ---")

for column in features + ["Final_Score"]:

    invalid = ((df[column] < 0) | (df[column] > 100)).sum()

    print(f"{column}: {invalid} invalid values")
# ============================================================
# STEP 3: TARGET ANALYSIS
# ============================================================

print("\n" + "=" * 50)
print("TARGET ANALYSIS")
print("=" * 50)

# Create temporary performance categories
def classify_performance(score):
    if score < 60:
        return "Needs Improvement"
    elif score < 80:
        return "Average Performer"
    else:
        return "High Performer"


df["Performance"] = df["Final_Score"].apply(classify_performance)

# Count each class
class_counts = df["Performance"].value_counts()

print("\nPerformance class counts:")
print(class_counts)

# Percentage of each class
class_percentages = df["Performance"].value_counts(normalize=True) * 100

print("\nPerformance class percentages:")
print(class_percentages.round(2))

# Check minimum and maximum Final Score in each class
print("\nFinal Score range by class:")
print(
    df.groupby("Performance")["Final_Score"]
      .agg(["count", "min", "max", "mean"])
      .round(2)
)
# ============================================================
# STEP 4: PREPARE DATA FOR MACHINE LEARNING
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os
import joblib


print("\n" + "=" * 50)
print("PREPARING DATA FOR MACHINE LEARNING")
print("=" * 50)


# ------------------------------------------------------------
# 4.1 SELECT OUR FOUR FEATURES
# ------------------------------------------------------------

features = [
    "Attendance (%)",
    "Midterm_Score",
    "Assignments_Avg",
    "Quizzes_Avg"
]

X = df[features].copy()
y = df["Performance"].copy()


print("\nFeatures used by ML model:")
for feature in features:
    print("-", feature)


# ------------------------------------------------------------
# 4.2 TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n--- TRAIN / TEST SPLIT ---")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# 4.3 CHECK CLASS DISTRIBUTION
# ------------------------------------------------------------

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# ------------------------------------------------------------
# 4.4 HANDLE MISSING VALUES
# ------------------------------------------------------------

print("\n--- MISSING VALUES BEFORE IMPUTATION ---")

print("Training:")
print(X_train.isnull().sum())

print("\nTesting:")
print(X_test.isnull().sum())


# Median imputation
# IMPORTANT: fit only on training data

imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


# Convert back to DataFrame

X_train_imputed = pd.DataFrame(
    X_train_imputed,
    columns=features
)

X_test_imputed = pd.DataFrame(
    X_test_imputed,
    columns=features
)


print("\n--- MISSING VALUES AFTER IMPUTATION ---")

print("Training missing values:")
print(X_train_imputed.isnull().sum().sum())

print("Testing missing values:")
print(X_test_imputed.isnull().sum().sum())


# ------------------------------------------------------------
# 4.5 STANDARDIZATION
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)


X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=features
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=features
)


print("\n--- STANDARDIZATION COMPLETE ---")

print("Training feature means:")
print(X_train_scaled.mean().round(4))

print("\nTraining feature standard deviations:")
print(X_train_scaled.std().round(4))


# ------------------------------------------------------------
# 4.6 CREATE OUTPUT FOLDERS
# ------------------------------------------------------------

os.makedirs("frozen_data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ------------------------------------------------------------
# 4.7 SAVE FROZEN DATA
# ------------------------------------------------------------

X_train_scaled.to_csv(
    "frozen_data/X_train_FROZEN.csv",
    index=False
)

X_test_scaled.to_csv(
    "frozen_data/X_test_FROZEN.csv",
    index=False
)

y_train.to_csv(
    "frozen_data/y_train_FROZEN.csv",
    index=False
)

y_test.to_csv(
    "frozen_data/y_test_FROZEN.csv",
    index=False
)


# ------------------------------------------------------------
# 4.8 SAVE PREPROCESSING OBJECTS
# ------------------------------------------------------------

joblib.dump(
    imputer,
    "models/imputer_FROZEN.pkl"
)

joblib.dump(
    scaler,
    "models/scaler_FROZEN.pkl"
)


# ------------------------------------------------------------
# 4.9 FINAL VERIFICATION
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("PREPROCESSING COMPLETE")
print("=" * 50)

print("\nX_train shape:", X_train_scaled.shape)
print("X_test shape :", X_test_scaled.shape)

print("y_train size:", len(y_train))
print("y_test size :", len(y_test))

print("\nFrozen files created:")
print("frozen_data/X_train_FROZEN.csv")
print("frozen_data/X_test_FROZEN.csv")
print("frozen_data/y_train_FROZEN.csv")
print("frozen_data/y_test_FROZEN.csv")

print("\nPreprocessing objects created:")
print("models/imputer_FROZEN.pkl")
print("models/scaler_FROZEN.pkl")
# ============================================================
# STEP 5: FEATURE-TARGET RELATIONSHIP ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("FEATURE-TARGET RELATIONSHIP ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# Correlation with Final Score
# ------------------------------------------------------------

print("\nCorrelation with Final Score:")

correlations = df[features + ["Final_Score"]].corr()["Final_Score"]

print(
    correlations
    .sort_values(ascending=False)
    .round(4)
)


# ------------------------------------------------------------
# Mean feature values for each performance class
# ------------------------------------------------------------

print("\nMean feature values by Performance class:")

class_means = df.groupby("Performance")[features].mean()

print(class_means.round(2))


# ------------------------------------------------------------
# Final Score vs selected features
# ------------------------------------------------------------

print("\nCorrelation matrix:")

print(
    df[features + ["Final_Score"]]
    .corr()
    .round(3)
)