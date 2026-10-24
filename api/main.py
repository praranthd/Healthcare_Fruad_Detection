
import logging
import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "data_transformation"
    / "preprocessor.joblib"
)

MLFLOW_TRACKING_URI = (
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)

MODEL_NAME = "HealthcareFraudDetectionModel"
MODEL_ALIAS = "champion"

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("healthcare_fraud_detection")


# --------------------------------------------------
# 2. Global model and preprocessor variables
# --------------------------------------------------

model = None
preprocessor = None
EXPECTED_COLUMNS = []


# --------------------------------------------------
# 3. Expected model input columns
# --------------------------------------------------

EXPECTED_COLUMNS = [
    "Patient_Age",
    "Procedure_Code",
    "Claim_Amount",
    "Days_Between_Service_and_Claim",
    "Number_of_Claims_Per_Provider_Monthly",
    "Length_of_Stay",
    "Chronic_Condition_Flag",
    "Prior_Visits_12m",
    "Claim_Submission_Date_month",
    "Claim_Submission_Date_day_of_week",
    "Patient_Gender",
    "Diagnosis_Code",
    "Insurance_Type",
    "Provider_Specialty",
    "Patient_State",
    "Visit_Type",
]


# --------------------------------------------------
# 4. Load model and preprocessor during startup
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocessor, EXPECTED_COLUMNS

    try:
        # Load the fitted preprocessing pipeline.
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(
                f"Preprocessor not found: {PREPROCESSOR_PATH}"
            )

        preprocessor = joblib.load(PREPROCESSOR_PATH)

        logger.info("Preprocessor loaded successfully.")

        # The preprocessor expects these raw columns after
        # the date has been converted into month and weekday.
        raw_columns = list(preprocessor.feature_names_in_)

        EXPECTED_COLUMNS = raw_columns

        logger.info("Expected columns: %s", EXPECTED_COLUMNS)

        # Connect MLflow to the local tracking database.
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Load the registered model using its champion alias.
        model = mlflow.sklearn.load_model(MODEL_URI)

        logger.info(
            "Champion model loaded successfully: %s",
            MODEL_URI,
        )

        # Verify that model and preprocessor output columns match.
        if hasattr(model, "feature_names_in_"):
            expected_model_features = list(model.feature_names_in_)
            actual_features = list(
                preprocessor.get_feature_names_out()
            )

            if expected_model_features != actual_features:
                raise ValueError(
                    "The model and preprocessor feature names "
                    "or their order do not match."
                )

        yield

    except Exception:
        logger.exception("Application startup failed.")
        raise

    finally:
        logger.info("Application shutting down.")


# --------------------------------------------------
# 5. Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Healthcare Fraud Detection API",
    description=(
        "An ML-powered API that estimates the probability "
        "that a healthcare claim is fraudulent."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# 6. Input schema
# --------------------------------------------------

class ClaimInput(BaseModel):
    Patient_Age: int = Field(ge=0, le=120)

    Patient_Gender: str
    Diagnosis_Code: str

    Procedure_Code: int
    Claim_Amount: float = Field(ge=0)

    Insurance_Type: str
    Provider_Specialty: str
    Patient_State: str
    Visit_Type: str

    Days_Between_Service_and_Claim: int = Field(ge=0)

    Number_of_Claims_Per_Provider_Monthly: int = Field(ge=0)

    Length_of_Stay: int = Field(ge=0)

    Chronic_Condition_Flag: int = Field(ge=0, le=1)

    Prior_Visits_12m: Optional[int] = Field(default=None, ge=0)

    Claim_Submission_Date: date


# --------------------------------------------------
# 7. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Healthcare Fraud Detection API is running.",
        "docs": "/docs",
        "health": "/health",
        "prediction_endpoint": "/predict",
    }


# --------------------------------------------------
# 8. Health-check endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    model_loaded = model is not None
    preprocessor_loaded = preprocessor is not None

    return {
        "status": (
            "healthy"
            if model_loaded and preprocessor_loaded
            else "unhealthy"
        ),
        "model_loaded": model_loaded,
        "preprocessor_loaded": preprocessor_loaded,
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
    }


# --------------------------------------------------
# 9. Prepare claim data for the model
# --------------------------------------------------

def transform_claim(claim: ClaimInput) -> pd.DataFrame:
    """
    Convert a single claim into the feature format expected
    by the fitted preprocessor and registered ML model.
    """

    if preprocessor is None:
        raise RuntimeError("Preprocessor is not loaded.")

    # Convert the Pydantic object into a dictionary.
    data = pd.DataFrame([claim.model_dump()])

    # Convert the submission date into a datetime value.
    data["Claim_Submission_Date"] = pd.to_datetime(
        data["Claim_Submission_Date"],
        errors="raise",
    )

    # Create the same date features used during training.
    data["Claim_Submission_Date_month"] = (
        data["Claim_Submission_Date"].dt.month
    )

    data["Claim_Submission_Date_day_of_week"] = (
        data["Claim_Submission_Date"].dt.dayofweek
    )

    # Remove the original date column because the fitted
    # preprocessor expects the engineered date features.
    data = data.drop(columns=["Claim_Submission_Date"])

    # Check that every expected input feature is present.
    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Input data is missing required columns: "
            f"{missing_columns}. "
            f"Available columns: {list(data.columns)}"
        )

    # Use exactly the training column order.
    data = data[EXPECTED_COLUMNS]

    # Apply the same fitted transformations used during training.
    transformed = preprocessor.transform(data)

    # Handle sparse output, if the preprocessor returns it.
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    # FIX: Restore the transformed feature names.
    # This matches the DataFrame feature names used to train
    # the registered Random Forest model.
    transformed = pd.DataFrame(
        transformed,
        columns=preprocessor.get_feature_names_out(),
        index=data.index,
    )

    return transformed


# --------------------------------------------------
# 10. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_claim(claim: ClaimInput):
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor is not loaded.",
        )

    try:
        # Step 1: Prepare the claim.
        transformed_claim = transform_claim(claim)

        # Step 2: Predict the class.
        prediction = int(model.predict(transformed_claim)[0])

        # Step 3: Calculate fraud probability.
        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(
                transformed_claim
            )

            classes = list(model.classes_)

            if 1 in classes:
                fraud_index = classes.index(1)
                probability = float(
                    probabilities[0][fraud_index]
                )

        # Step 4: Prepare a readable result.
        is_fraud = prediction == 1

        result = (
            "Potential fraud detected"
            if is_fraud
            else "No fraud detected by the model"
        )

        message = (
            "This claim has been flagged for further review. "
            "The prediction is not proof of fraud."
            if is_fraud
            else
            "The model did not flag this claim. "
            "This does not guarantee that the claim is legitimate."
        )

        # Step 5: Return a structured response.
        return {
            "prediction": prediction,
            "is_fraud": prediction,
            "fraud_probability": probability,
            "result": result,
            "message": message,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc
