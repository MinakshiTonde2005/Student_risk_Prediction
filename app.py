import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "Logistic.pkl"

st.set_page_config(
    page_title="Student Risk Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7ff 0%, #eef2ff 45%, #ffffff 100%);
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: #172554;
    margin-bottom: 5px;
}
.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 17px;
    margin-bottom: 28px;
}
.card {
    background: rgba(255,255,255,0.96);
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0 8px 28px rgba(15,23,42,0.10);
    border: 1px solid #e2e8f0;
    margin-bottom: 20px;
}
.result-card {
    padding: 25px;
    border-radius: 18px;
    background: white;
    box-shadow: 0 10px 30px rgba(15,23,42,0.12);
    text-align: center;
    border: 1px solid #e2e8f0;
}
.result-title {
    font-size: 20px;
    color: #475569;
}
.result-value {
    font-size: 36px;
    font-weight: 800;
    color: #1d4ed8;
}
div.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 17px;
    font-weight: 700;
}
section[data-testid="stSidebar"] {
    background: #f8fafc;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error("Could not load Logistic.pkl. Keep Logistic.pkl in the same folder as app.py.")
    st.stop()

# -------------------- Header --------------------
st.markdown('<div class="main-title">📊 Student Risk Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Logistic Regression based prediction dashboard</div>',
    unsafe_allow_html=True
)

st.info("Enter the student details below and click Predict.")

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Model Information")
    st.write("**Model:** Logistic Regression")
    st.write("**Features:** 10")
    st.write("**Classes:** At-Risk, High-Risk, Safe")
    st.markdown("---")
    st.caption("Logistic.pkl must be in the same folder as app.py.")

# -------------------- Inputs --------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📝 Student Information")

col1, col2 = st.columns(2)

with col1:
    attendance = st.number_input(
        "Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0
    )

    study_hours = st.number_input(
        "Study Hours", min_value=0.0, value=3.0, step=0.5
    )

    past_failures = st.number_input(
        "Past Failures", min_value=0, value=0, step=1
    )

    assignments_completed_pct = st.number_input(
        "Assignments Completed (%)",
        min_value=0.0, max_value=100.0, value=80.0, step=1.0
    )

    # Categorical field.
    # IMPORTANT: The numeric codes must match the encoding used during training.
    parental_education_label = st.selectbox(
        "Parental Education",
        ["0 - Category 0", "1 - Category 1", "2 - Category 2", "3 - Category 3"]
    )
    parental_education = int(parental_education_label.split(" - ")[0])

with col2:
    family_income = st.number_input(
        "Family Income", min_value=0.0, value=50000.0, step=1000.0
    )

    extracurricular_label = st.selectbox(
        "Extracurricular Activity", ["No", "Yes"]
    )
    extracurricular = 1 if extracurricular_label == "Yes" else 0

    internet_access_label = st.selectbox(
        "Internet Access", ["No", "Yes"]
    )
    internet_access = 1 if internet_access_label == "Yes" else 0

    previous_grade = st.number_input(
        "Previous Grade", min_value=0.0, max_value=100.0, value=65.0, step=1.0
    )

    final_score = st.number_input(
        "Final Score", min_value=0.0, max_value=100.0, value=65.0, step=1.0
    )

st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Prediction --------------------
if st.button("🚀 Predict Student Risk"):
    input_data = pd.DataFrame([{
        "attendance": attendance,
        "study_hours": study_hours,
        "past_failures": past_failures,
        "assignments_completed_pct": assignments_completed_pct,
        "parental_education": parental_education,
        "family_income": family_income,
        "extracurricular": extracurricular,
        "internet_access": internet_access,
        "previous_grade": previous_grade,
        "final_score": final_score
    }])

    try:
        expected_features = list(
            getattr(model, "feature_names_in_", input_data.columns)
        )
        input_data = input_data[expected_features]

        prediction = model.predict(input_data)[0]

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">Predicted Student Category</div>
                <div class="result-value">{prediction}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_

            st.subheader("📈 Prediction Probabilities")

            prob_df = pd.DataFrame({
                "Category": classes,
                "Probability": probabilities
            })
            prob_df["Probability"] = prob_df["Probability"].round(4)

            st.bar_chart(prob_df.set_index("Category")["Probability"])
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Built with Streamlit • Logistic Regression")
