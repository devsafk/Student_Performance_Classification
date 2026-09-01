# ============================================================
# 04_prediction.py
# STUDENT PERFORMANCE CLASSIFICATION
# STUDENT PREDICTION SYSTEM
# ============================================================

import os
import joblib
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/best_model_improved.pkl"

FEATURES = [
    "Attendance (%)",
    "Midterm_Score",
    "Assignments_Avg",
    "Quizzes_Avg"
]

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STUDENT PERFORMANCE CLASSIFICATION")
print("STUDENT PREDICTION SYSTEM")
print("=" * 70)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Trained model not found.")
    print(f"Expected file: {MODEL_PATH}")
    print("\nMake sure you have already run:")
    print("02_train_models.py")
    exit()

try:
    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")
    print(f"Model type: {type(model).__name__}")

except Exception as e:
    print("\nERROR while loading model:")
    print(e)
    exit()

# ============================================================
# FEATURES USED
# ============================================================

print("\n" + "=" * 70)
print("FEATURES USED")
print("=" * 70)

for i, feature in enumerate(FEATURES, 1):
    print(f"{i}. {feature}")

# ============================================================
# INPUT VALIDATION FUNCTION
# ============================================================

def get_score(prompt):
    while True:
        try:
            value = float(input(prompt))

            if 0 <= value <= 100:
                return value

            print("Please enter a value between 0 and 100.")

        except ValueError:
            print("Invalid input. Please enter a numeric value.")


# ============================================================
# STUDENT INPUT
# ============================================================

print("\n" + "=" * 70)
print("ENTER STUDENT DETAILS")
print("=" * 70)

attendance = get_score(
    "\nEnter Attendance (%) [0-100]: "
)

midterm = get_score(
    "Enter Midterm Score [0-100]: "
)

assignments = get_score(
    "Enter Assignments Average [0-100]: "
)

quizzes = get_score(
    "Enter Quizzes Average [0-100]: "
)

# ============================================================
# CREATE INPUT
# ============================================================

student_data = np.array([
    [attendance, midterm, assignments, quizzes]
])

# ============================================================
# GENERATE PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTION")
print("=" * 70)

try:
    prediction = model.predict(student_data)[0]

    probabilities = model.predict_proba(student_data)[0]

except Exception as e:
    print("\nERROR while generating prediction:")
    print(e)
    exit()

# ============================================================
# PREDICTION RESULT
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

print(f"\nPredicted Performance: {prediction}")

# ============================================================
# PREDICTION PROBABILITIES
# ============================================================

print("\nPrediction probabilities:")

classes = model.classes_

for class_name, probability in zip(classes, probabilities):
    print(f"{class_name:<25} {probability * 100:.2f}%")

# ============================================================
# STUDENT INPUT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STUDENT INPUT SUMMARY")
print("=" * 70)

print(f"\nAttendance       : {attendance:.2f}%")
print(f"Midterm Score    : {midterm:.2f}")
print(f"Assignments Avg  : {assignments:.2f}")
print(f"Quizzes Avg      : {quizzes:.2f}")

# ============================================================
# IMPROVEMENT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("IMPROVEMENT ANALYSIS")
print("=" * 70)

scores = {
    "Attendance": attendance,
    "Midterm Score": midterm,
    "Assignments": assignments,
    "Quizzes": quizzes
}

# ------------------------------------------------------------
# Score interpretation
# ------------------------------------------------------------

def interpret_score(score):

    if score >= 85:
        return "Strong"

    elif score >= 75:
        return "Good"

    elif score >= 60:
        return "Moderate"

    else:
        return "Needs Attention"


print()

for feature, score in scores.items():
    status = interpret_score(score)

    print(f"{feature:<18}: {score:>6.2f}%  -> {status}")

# ============================================================
# IDENTIFY AREAS FOR IMPROVEMENT
# ============================================================

# Features below 75 are considered improvement areas.
# If none are below 75, identify the weakest feature(s).

IMPROVEMENT_THRESHOLD = 75

improvement_areas = {
    feature: score
    for feature, score in scores.items()
    if score < IMPROVEMENT_THRESHOLD
}

# ------------------------------------------------------------
# If no feature is below threshold, identify weakest feature
# ------------------------------------------------------------

if len(improvement_areas) == 0:

    weakest_score = min(scores.values())

    improvement_areas = {
        feature: score
        for feature, score in scores.items()
        if score == weakest_score
    }

# ============================================================
# RECOMMENDATIONS
# ============================================================

print("\n" + "-" * 70)
print("AREAS FOR IMPROVEMENT")
print("-" * 70)

recommendations = {
    "Attendance":
        "Maintain regular attendance and avoid unnecessary absences.",

    "Midterm Score":
        "Focus on revision, concept clarity and regular test practice.",

    "Assignments":
        "Complete assignments consistently and improve accuracy.",

    "Quizzes":
        "Practice short quizzes regularly to strengthen topic retention."
}

for feature, score in sorted(
    improvement_areas.items(),
    key=lambda x: x[1]
):

    print(f"\n• {feature}: {score:.2f}%")
    print(f"  Recommendation: {recommendations[feature]}")

# ============================================================
# OVERALL PERFORMANCE SCORE
# ============================================================

overall_score = np.mean([
    attendance,
    midterm,
    assignments,
    quizzes
])

print("\n" + "=" * 70)
print("OVERALL ACADEMIC INDICATOR")
print("=" * 70)

print(f"\nAverage of input scores: {overall_score:.2f}%")

if overall_score >= 85:
    overall_status = "Strong overall performance"

elif overall_score >= 75:
    overall_status = "Good overall performance"

elif overall_score >= 60:
    overall_status = "Moderate overall performance"

else:
    overall_status = "Performance needs attention"

print(f"Status: {overall_status}")

# ============================================================
# INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

if prediction == "High Performer":

    print(
        "\nThe model predicts the student as a High Performer."
    )

elif prediction == "Average Performer":

    print(
        "\nThe model predicts the student as an Average Performer."
    )

else:

    print(
        "\nThe model predicts that the student Needs Improvement."
    )

print(
    "\nNote: Model probability represents classification confidence."
)

print(
    "The improvement analysis separately identifies areas where"
    " the student's scores can be strengthened."
)

# ============================================================
# FINAL RECOMMENDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL RECOMMENDATION")
print("=" * 70)

if improvement_areas:

    best_area = min(
        improvement_areas,
        key=improvement_areas.get
    )

    print(
        f"\nPrimary focus area: {best_area}"
    )

    print(
        "Improving this area while maintaining current performance "
        "can help the student progress toward the next performance level."
    )

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION COMPLETE")
print("=" * 70)