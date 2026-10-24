
import os
from datetime import date

import requests
import streamlit as st

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="FraudGuard | Healthcare Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


# --------------------------------------------------
# 2. CUSTOM CSS
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #f4f7fb;
    }

    [data-testid="stHeader"] {
        background: rgba(244, 247, 251, 0.92);
    }

    [data-testid="stSidebar"] {
        background: #111827;
    }

    [data-testid="stSidebar"] * {
        color: #f3f4f6;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: transparent;
        border-radius: 10px;
        padding: 8px 10px;
    }

    .brand {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -1px;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .brand span {
        color: #5eead4;
    }

    .sidebar-caption {
        color: #9ca3af;
        font-size: 12px;
        margin-bottom: 28px;
    }

    .hero {
        background: linear-gradient(120deg, #102a43, #155e75);
        padding: 30px;
        border-radius: 20px;
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        font-size: 32px;
        font-weight: 800;
        margin: 0 0 10px 0;
    }

    .hero p {
        color: #dbeafe;
        font-size: 14px;
        margin: 0;
    }

    .section-title {
        font-size: 21px;
        font-weight: 750;
        color: #172b4d;
        margin: 20px 0 5px 0;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 13px;
        margin-bottom: 18px;
    }

    .metric-card {
        background: white;
        border: 1px solid #e5eaf2;
        border-radius: 16px;
        padding: 20px;
        min-height: 132px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.03);
    }

    .metric-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 600;
    }

    .metric-value {
        color: #172b4d;
        font-size: 29px;
        font-weight: 800;
        margin: 10px 0 4px 0;
    }

    .metric-note {
        color: #64748b;
        font-size: 12px;
    }

    .panel {
        background: white;
        border: 1px solid #e5eaf2;
        border-radius: 16px;
        padding: 22px;
        margin: 14px 0;
    }

    .result-fraud {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        border-left: 6px solid #e11d48;
        border-radius: 16px;
        padding: 22px;
        margin-top: 20px;
    }

    .result-safe {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-left: 6px solid #059669;
        border-radius: 16px;
        padding: 22px;
        margin-top: 20px;
    }

    .result-title {
        font-size: 23px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button[kind="primary"] {
        background: #0f766e;
        border: none;
        border-radius: 10px;
        min-height: 46px;
        font-weight: 700;
        color: white;
        transition: 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover,
    div.stFormSubmitButton > button[kind="primary"]:hover {
        background: #115e59;
        border: none;
        transform: translateY(-1px);
    }

    div[data-testid="stForm"] {
        background: white;
        border: 1px solid #e5eaf2;
        border-radius: 16px;
        padding: 22px;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5eaf2;
        border-radius: 14px;
        padding: 16px;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        margin-top: 45px;
        padding: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. REUSABLE FUNCTIONS
# --------------------------------------------------
def get_api_status():
    """Check whether the FastAPI server is responding."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)

        if response.status_code == 200:
            return response.json()

        return {"status": "unavailable"}

    except requests.RequestException:
        return {"status": "offline"}


def metric_card(label, value, note):
    """Display a dashboard metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_prediction_result(result):
    """Display the result returned by FastAPI."""
    prediction = int(result["prediction"])
    probability = result.get("fraud_probability")

    if prediction == 1:
        st.markdown(
            """
            <div class="result-fraud">
                <div class="result-title">
                    ⚠️ Potential Fraud Flag
                </div>
                <div>
                    The model classified this claim as potentially fraudulent.
                    Review the claim before making a decision.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-safe">
                <div class="result-title">
                    ✓ No Fraud Flag
                </div>
                <div>
                    The model did not flag this claim as fraudulent.
                    This does not guarantee that the claim is legitimate.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if probability is not None:
        probability = max(0.0, min(1.0, float(probability)))

        st.markdown("### Model confidence indicator")
        st.progress(probability)
        st.write(f"**Estimated fraud probability:** {probability:.2%}")

    st.caption(
        "This is a machine-learning screening result, not a confirmed "
        "finding of fraud. Human review is required."
    )


# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">Fraud<span>Guard</span></div>
        <div class="sidebar-caption">
            HEALTHCARE INTELLIGENCE PLATFORM
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Fraud Prediction", "About"],
        label_visibility="visible",
    )

    st.markdown("---")
    st.caption("SYSTEM CONNECTION")

    if st.button("Check API status", use_container_width=True):
        st.session_state["api_status"] = get_api_status()

    api_status = st.session_state.get(
        "api_status",
        get_api_status(),
    )

    if (
        api_status.get("status") == "healthy"
        and api_status.get("model_loaded")
        and api_status.get("preprocessor_loaded")
    ):
        st.success("API and model connected")
    elif api_status.get("status") == "healthy":
        st.warning("API running; model status needs checking")
    else:
        st.error("API offline or unavailable")

    st.markdown("---")
    st.caption("Local development environment")


# --------------------------------------------------
# 5. DASHBOARD PAGE
# --------------------------------------------------
if page == "Dashboard":

    st.markdown(
        """
        <div class="hero">
            <h1>Healthcare Fraud Intelligence</h1>
            <p>
                Monitor model performance and screen healthcare claims
                using your machine-learning pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Model performance overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Validation results from your current Random Forest model.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("F1 SCORE", "66.49%", "Validation set")

    with col2:
        metric_card("RECALL", "75.30%", "Fraud cases detected")

    with col3:
        metric_card("ROC-AUC", "96.49%", "Validation set")

    with col4:
        metric_card("PRECISION", "59.52%", "Predicted fraud cases")

    st.markdown(
        '<div class="section-title">System overview</div>',
        unsafe_allow_html=True,
    )

    status = get_api_status()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("API service")

        if status.get("status") == "healthy":
            st.success("FastAPI is responding")
        else:
            st.error("FastAPI is not responding")

        st.caption(f"Endpoint: {API_URL}/predict")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Registered model")

        if status.get("model_loaded"):
            st.success("Model loaded")
        else:
            st.warning("Model not confirmed as loaded")

        if status.get("preprocessor_loaded"):
            st.success("Preprocessor loaded")
        else:
            st.warning("Preprocessor not confirmed as loaded")

        st.caption("Model source: MLflow champion alias")
        st.markdown("</div>", unsafe_allow_html=True)

    st.info(
        "Choose **Fraud Prediction** in the sidebar to evaluate an individual claim."
    )

    st.caption(
        "Metrics are from the current validation results. "
        "They are not live counts of claims processed."
    )


# --------------------------------------------------
# 6. FRAUD PREDICTION PAGE
# --------------------------------------------------
elif page == "Fraud Prediction":

    st.markdown(
        """
        <div class="hero">
            <h1>Claim Risk Assessment</h1>
            <p>
                Enter the claim information below to request a prediction
                from your trained model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Claim information</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Complete the required fields. Use the same category values and '
        'units as your training dataset.'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.form("claim_prediction_form"):

        st.markdown("#### 01 · Patient details")

        c1, c2 = st.columns(2)

        with c1:
            patient_age = st.number_input(
                "Patient age",
                min_value=0,
                max_value=120,
                value=45,
            )

            patient_gender = st.text_input(
                "Patient gender",
                value="Female",
                help="Use a category that exists in your training dataset.",
            )

            patient_state = st.text_input(
                "Patient state",
                value="CA",
            )

        with c2:
            diagnosis_code = st.text_input(
                "Diagnosis code",
                value="I10",
            )

            chronic_condition = st.selectbox(
                "Chronic condition",
                options=["No", "Yes"],
            )

            prior_visits = st.number_input(
                "Prior visits in the last 12 months",
                min_value=0,
                value=2,
            )

        st.markdown("---")
        st.markdown("#### 02 · Claim and treatment details")

        c1, c2 = st.columns(2)

        with c1:
            procedure_code = st.number_input(
                "Procedure code",
                min_value=0,
                value=120,
                step=1,
            )

            claim_amount = st.number_input(
                "Claim amount",
                min_value=0.0,
                value=1500.0,
                step=100.0,
            )

            insurance_type = st.text_input(
                "Insurance type",
                value="Private",
                help="Enter a category used by the training dataset.",
            )

            visit_type = st.text_input(
                "Visit type",
                value="Outpatient",
            )

        with c2:
            provider_specialty = st.text_input(
                "Provider specialty",
                value="Cardiology",
            )

            days_between = st.number_input(
                "Days between service and claim",
                min_value=0,
                value=5,
            )

            monthly_claims = st.number_input(
                "Monthly claims per provider",
                min_value=0,
                value=20,
            )

            length_of_stay = st.number_input(
                "Length of stay (days)",
                min_value=0,
                value=1,
            )

        st.markdown("---")
        st.markdown("#### 03 · Submission details")

        submission_date = st.date_input(
            "Claim submission date",
            value=date.today(),
        )

        submitted = st.form_submit_button(
            "Run fraud risk assessment",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        payload = {
            "Patient_Age": int(patient_age),
            "Patient_Gender": patient_gender.strip(),
            "Diagnosis_Code": diagnosis_code.strip(),
            "Procedure_Code": int(procedure_code),
            "Claim_Amount": float(claim_amount),
            "Insurance_Type": insurance_type.strip() or None,
            "Provider_Specialty": provider_specialty.strip() or None,
            "Patient_State": patient_state.strip(),
            "Visit_Type": visit_type.strip(),
            "Days_Between_Service_and_Claim": int(days_between),
            "Number_of_Claims_Per_Provider_Monthly": int(monthly_claims),
            "Length_of_Stay": int(length_of_stay),
            "Chronic_Condition_Flag": (
                1 if chronic_condition == "Yes" else 0
            ),
            "Prior_Visits_12m": int(prior_visits),
            "Claim_Submission_Date": submission_date.isoformat(),
        }

        try:
            with st.spinner("Analyzing claim with the trained model..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=30,
                )

            if response.status_code == 200:
                result = response.json()

                st.markdown("---")
                st.markdown(
                    '<div class="section-title">Assessment result</div>',
                    unsafe_allow_html=True,
                )

                show_prediction_result(result)

                with st.expander("View submitted claim details"):
                    st.json(payload)

                with st.expander("View API response"):
                    st.json(result)

            else:
                st.error(
                    f"Prediction failed (HTTP {response.status_code})."
                )

                try:
                    st.json(response.json())
                except ValueError:
                    st.code(response.text)

                st.caption(
                    "Check the FastAPI terminal for the detailed error. "
                    "Make sure the input categories and feature names match "
                    "your training pipeline."
                )

        except requests.RequestException as exc:
            st.error("Could not connect to the prediction API.")
            st.write(f"Details: {exc}")
            st.info(
                "Start FastAPI first, then try the prediction again."
            )


# --------------------------------------------------
# 7. ABOUT PAGE
# --------------------------------------------------
elif page == "About":

    st.markdown(
        """
        <div class="hero">
            <h1>About FraudGuard</h1>
            <p>
                A local machine-learning application for healthcare
                claim risk screening.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Technology stack")

    st.markdown(
        """
        - **Frontend:** Streamlit with custom CSS
        - **Backend:** FastAPI
        - **Machine learning:** Scikit-learn Random Forest
        - **Experiment tracking and model registry:** MLflow
        - **Data processing:** Pandas and scikit-learn pipelines
        """
    )

    st.markdown("### Prediction workflow")

    st.markdown(
        """
        1. Enter claim details in the Streamlit interface.
        2. Streamlit sends the data to the FastAPI `/predict` endpoint.
        3. FastAPI applies the saved preprocessing pipeline.
        4. The registered MLflow champion model predicts the outcome.
        5. Streamlit displays the prediction and estimated probability.
        """
    )

    st.warning(
        "This application supports claim screening. Its predictions "
        "should not be treated as proof of fraud or used as the sole "
        "basis for denying a claim."
    )


# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
        FraudGuard · Healthcare Fraud Detection · Local MLOps Project
    </div>
    """,
    unsafe_allow_html=True,
)
