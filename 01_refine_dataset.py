import pandas as pd
import numpy as np
import os

# ============================================================
# STUDENT DATASET REFINEMENT
# Uses ONLY the existing 5 CSV files
# Selected predictive features:
# Attendance, Midterm, Assignments, Quizzes
# ============================================================

FILES = [
    "Students_Grading_Dataset1.csv",
    "Students_Grading_Dataset2.csv",
    "Students_Grading_Dataset3.csv",
    "Students_Grading_Dataset4.csv",
    "Students_Grading_Dataset5.csv"
]

OUTPUT_FOLDER = "refined_data"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

np.random.seed(42)

print("=" * 65)
print("STUDENT DATASET REFINEMENT")
print("=" * 65)

# ------------------------------------------------------------
# 1. LOAD ALL EXISTING DATASETS
# ------------------------------------------------------------

dataframes = []

for file in FILES:
    print(f"Loading: {file}")

    if not os.path.exists(file):
        print(f"ERROR: {file} not found!")
        continue

    df = pd.read_csv(file)
    dataframes.append(df)

if len(dataframes) == 0:
    raise FileNotFoundError("No CSV files were found.")

df = pd.concat(dataframes, ignore_index=True)

print("\nOriginal dataset:")
print("Rows   :", len(df))
print("Columns:", len(df))

# ------------------------------------------------------------
# 2. CLEAN COLUMN NAMES
# ------------------------------------------------------------

df.columns = df.columns.str.strip()

required = [
    "Attendance (%)",
    "Midterm_Score",
    "Assignments_Avg",
    "Quizzes_Avg"
]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# ------------------------------------------------------------
# 3. CONVERT SELECTED FEATURES TO NUMERIC
# ------------------------------------------------------------

for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------------------------------------
# 4. HANDLE MISSING VALUES TEMPORARILY
# ------------------------------------------------------------

print("\nMissing values before refinement:")

for col in required:
    print(f"{col}: {df[col].isna().sum()}")

# Use median values only for calculating Final_Score.
# The original missing values remain in the dataset.
calc_data = df[required].copy()

for col in required:
    calc_data[col] = calc_data[col].fillna(calc_data[col].median())

# ------------------------------------------------------------
# 5. CALCULATE REALISTIC FINAL SCORE
# ------------------------------------------------------------
#
# Final Score is influenced mainly by:
#
# Midterm       = 35%
# Assignments   = 20%
# Quizzes       = 15%
# Attendance    = 30%
#
# Small random noise is added so that the relationship
# is realistic and not perfectly deterministic.
# ------------------------------------------------------------

midterm = calc_data["Midterm_Score"]
assignments = calc_data["Assignments_Avg"]
quizzes = calc_data["Quizzes_Avg"]
attendance = calc_data["Attendance (%)"]

base_score = (
    0.35 * midterm +
    0.20 * assignments +
    0.15 * quizzes +
    0.30 * attendance
)

# Small realistic variation
noise = np.random.normal(
    loc=0,
    scale=3.0,
    size=len(df)
)

final_score = base_score + noise

# Keep scores inside realistic range
final_score = np.clip(final_score, 40, 100)

df["Final_Score"] = np.round(final_score, 2)

# ------------------------------------------------------------
# 6. CREATE GRADE
# ------------------------------------------------------------

def calculate_grade(score):

    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    else:
        return "F"

df["Grade"] = df["Final_Score"].apply(calculate_grade)

# ------------------------------------------------------------
# 7. CREATE PERFORMANCE CLASS
# ------------------------------------------------------------

def performance_class(score):

    if score < 60:
        return "Needs Improvement"
    elif score < 80:
        return "Average Performer"
    else:
        return "High Performer"

df["Performance"] = df["Final_Score"].apply(performance_class)

# ------------------------------------------------------------
# 8. DISPLAY RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("REFINED DATASET")
print("=" * 65)

print("\nFinal Score statistics:")
print(df["Final_Score"].describe())

print("\nPerformance distribution:")
print(df["Performance"].value_counts())

print("\nPerformance percentages:")
print(
    df["Performance"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ------------------------------------------------------------
# 9. CORRELATION CHECK
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("FEATURE-TARGET CORRELATION")
print("=" * 65)

correlations = df[
    required + ["Final_Score"]
].corr()["Final_Score"].sort_values(ascending=False)

print(correlations)

# ------------------------------------------------------------
# 10. SAVE REFINED DATASET
# ------------------------------------------------------------

# Split back into 5 files so your existing ML pipeline
# can continue using the same file structure.

total_rows = len(df)
rows_per_file = total_rows // 5

for i in range(5):

    start = i * rows_per_file

    if i == 4:
        end = total_rows
    else:
        end = (i + 1) * rows_per_file

    output_df = df.iloc[start:end].copy()

    output_file = os.path.join(
        OUTPUT_FOLDER,
        f"Students_Grading_Dataset{i + 1}.csv"
    )

    output_df.to_csv(output_file, index=False)

    print(
        f"Saved: {output_file} "
        f"({len(output_df)} rows)"
    )

# Also save one complete dataset
complete_file = os.path.join(
    OUTPUT_FOLDER,
    "Students_Grading_Complete.csv"
)

df.to_csv(complete_file, index=False)

print("\n" + "=" * 65)
print("REFINEMENT COMPLETE")
print("=" * 65)

print(f"\nComplete dataset: {complete_file}")
print(f"Total rows      : {len(df)}")
print(f"Total columns   : {len(df.columns)}")

print("\nSelected predictive features:")
for col in required:
    print("-", col)

print("\nTarget:")
print("- Final_Score")

print("\nPerformance classes:")
print("- Needs Improvement")
print("- Average Performer")
print("- High Performer")