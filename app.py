import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Classification",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    /* Reduce vertical spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Smaller headings */
    h1 {
        margin-bottom: 0.5rem;
    }

    h2 {
        margin-top: 1.2rem;
        margin-bottom: 0.7rem;
    }

    h3 {
        margin-top: 1rem;
        margin-bottom: 0.6rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🎓 Student Performance Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Student Performance Prediction System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "best_model_improved.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):

    st.error(
        "❌ Trained model not found.\n\n"
        "Please make sure this file exists:\n"
        "`models/best_model_improved.pkl`"
    )

    st.stop()


try:

    model = load_model()

except Exception as e:

    st.error(f"❌ Error loading model: {e}")

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🤖 Model Information")

    st.write("**Model:** K-Nearest Neighbors (KNN)")
    st.write("**Task:** Multi-Class Classification")
    st.write("**Accuracy:** 86.90%")
    st.write("**Macro F1:** 0.7501")

    st.divider()

    st.header("📌 Input Features")

    st.write("• Attendance (%)")
    st.write("• Midterm Score")
    st.write("• Assignments Average")
    st.write("• Quizzes Average")

    st.divider()

    st.info(
        "The model predicts the student's performance "
        "category using four academic indicators."
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.header("📝 Enter Student Details")

st.write(
    "Enter the student's academic information below "
    "to generate a performance prediction."
)


col1, col2 = st.columns(2)


with col1:

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )

    midterm = st.number_input(
        "Midterm Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )


with col2:

    assignments = st.number_input(
        "Assignments Average",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )

    quizzes = st.number_input(
        "Quizzes Average",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )


# ============================================================
# INPUT DATA
# ============================================================

input_data = pd.DataFrame({

    "Attendance (%)": [attendance],

    "Midterm_Score": [midterm],

    "Assignments_Avg": [assignments],

    "Quizzes_Avg": [quizzes]

})


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Student Performance",
    use_container_width=True,
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]

        prediction = str(prediction)


        # ----------------------------------------------------
        # PREDICTION PROBABILITIES
        # ----------------------------------------------------

        probability_dict = {}

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            classes = model.classes_

            probability_dict = {
                str(cls): float(prob)
                for cls, prob in zip(classes, probabilities)
            }


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.header("📊 Prediction Result")


        # ----------------------------------------------------
        # RESULT MESSAGE
        # ----------------------------------------------------

        if prediction == "High Performer":

            result_heading = "🏆 HIGH PERFORMER"

            result_text = (
                "The student is performing at a high level."
            )

        elif prediction == "Needs Improvement":

            result_heading = "⚠️ NEEDS IMPROVEMENT"

            result_text = (
                "The student may benefit from additional "
                "academic support."
            )

        else:

            result_heading = "📘 AVERAGE PERFORMER"

            result_text = (
                "The student is performing at an average level."
            )


        # ----------------------------------------------------
        # CLEAN RESULT BOX
        # ----------------------------------------------------
        # IMPORTANT:
        # No custom HTML is used here.
        # This prevents raw <div> tags from appearing.

        with st.container(border=True):

            st.markdown(
                f"## {result_heading}"
            )

            st.write(
                result_text
            )


        # ====================================================
        # STUDENT INPUT SUMMARY
        # ====================================================

        st.subheader("📋 Student Input Summary")


        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "Attendance",
                f"{attendance:.0f}%"
            )


        with c2:

            st.metric(
                "Midterm",
                f"{midterm:.0f}"
            )


        with c3:

            st.metric(
                "Assignments",
                f"{assignments:.0f}"
            )


        with c4:

            st.metric(
                "Quizzes",
                f"{quizzes:.0f}"
            )


        # ====================================================
        # PREDICTION PROBABILITIES
        # ====================================================

        if probability_dict:

            st.subheader("📈 Prediction Probabilities")


            probability_df = pd.DataFrame({

                "Performance": list(
                    probability_dict.keys()
                ),

                "Probability (%)": [
                    value * 100
                    for value in probability_dict.values()
                ]

            })


            probability_df = probability_df.sort_values(
                "Probability (%)",
                ascending=False
            )


            # ------------------------------------------------
            # SMALL PROBABILITY CHART
            # ------------------------------------------------

            st.bar_chart(
                probability_df.set_index("Performance"),
                y="Probability (%)",
                height=220
            )


            # ------------------------------------------------
            # EXACT PROBABILITY VALUES
            # ------------------------------------------------

            st.write("**Probability Values**")


            pc1, pc2, pc3 = st.columns(3)


            probability_values = {
                row["Performance"]:
                row["Probability (%)"]
                for _, row in probability_df.iterrows()
            }


            with pc1:

                st.metric(
                    "Average Performer",
                    f"{probability_values.get('Average Performer', 0):.2f}%"
                )


            with pc2:

                st.metric(
                    "High Performer",
                    f"{probability_values.get('High Performer', 0):.2f}%"
                )


            with pc3:

                st.metric(
                    "Needs Improvement",
                    f"{probability_values.get('Needs Improvement', 0):.2f}%"
                )


        # ====================================================
        # STUDENT PERFORMANCE INDICATORS
        # ====================================================

        st.subheader("📊 Student Performance Indicators")


        performance_df = pd.DataFrame({

            "Indicator": [
                "Attendance",
                "Midterm",
                "Assignments",
                "Quizzes"
            ],

            "Score": [
                attendance,
                midterm,
                assignments,
                quizzes
            ]

        })


        # ----------------------------------------------------
        # SMALL PERFORMANCE CHART
        # ----------------------------------------------------

        st.bar_chart(
            performance_df.set_index("Indicator"),
            y="Score",
            height=250
        )


        # ====================================================
        # PERFORMANCE ANALYSIS
        # ====================================================

        st.subheader("💡 Performance Analysis")


        scores = {

            "Attendance": attendance,

            "Midterm": midterm,

            "Assignments": assignments,

            "Quizzes": quizzes

        }


        highest_feature = max(
            scores,
            key=scores.get
        )

        highest_score = scores[highest_feature]


        lowest_feature = min(
            scores,
            key=scores.get
        )

        lowest_score = scores[lowest_feature]


        st.write(
            f"**Strongest area:** "
            f"{highest_feature} "
            f"({highest_score:.0f}/100)"
        )


        st.write(
            f"**Area needing the most attention:** "
            f"{lowest_feature} "
            f"({lowest_score:.0f}/100)"
        )


        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        st.subheader("🎯 Recommendations")


        recommendations = []


        if attendance < 75:

            recommendations.append(
                "Improve class attendance and maintain regular participation."
            )


        if midterm < 60:

            recommendations.append(
                "Focus on improving midterm preparation and conceptual understanding."
            )


        if assignments < 60:

            recommendations.append(
                "Complete assignments regularly and improve assignment performance."
            )


        if quizzes < 60:

            recommendations.append(
                "Practice more frequently to improve quiz performance."
            )


        if not recommendations:

            recommendations.append(
                "Maintain the current academic performance "
                "and continue consistent study habits."
            )


        for recommendation in recommendations:

            st.write(
                f"• {recommendation}"
            )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.divider()

        st.caption(
            "Note: This prediction is generated by a machine "
            "learning model and should be used as an academic "
            "support tool, not as the sole basis for evaluating "
            "a student."
        )


    except Exception as e:

        st.error(
            f"❌ Prediction error: {e}"
        )