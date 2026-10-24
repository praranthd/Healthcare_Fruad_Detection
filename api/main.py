
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. Configuration
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


# --------------------------------------------------
# 2. Logging
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("api.main")


# --------------------------------------------------
# 3. Global model objects
# --------------------------------------------------

preprocessor = None
model = None
EXPECTED_COLUMNS = None


# --------------------------------------------------
# 4. Input schema
# --------------------------------------------------

class ClaimInput(BaseModel):
    Patient_Age: int = Field(ge=0, le=120)
    Patient_Gender: str
    Diagnosis_Code: str
    Procedure_Code: int
    Claim_Amount: float = Field(ge=0)
    Insurance_Type: Optional[str] = None
    Provider_Specialty: Optional[str] = None
    Patient_State: str
    Visit_Type: str

    Days_Between_Service_and_Claim: int = Field(ge=0)
    Number_of_Claims_Per_Provider_Monthly: int = Field(ge=0)
    Length_of_Stay: int = Field(ge=0)
    Chronic_Condition_Flag: int = Field(ge=0, le=1)
    Prior_Visits_12m: Optional[int] = Field(default=None, ge=0)

    Claim_Submission_Date: str


# --------------------------------------------------
# 5. Load the model and preprocessor
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global preprocessor, model, EXPECTED_COLUMNS

    logger.info("Loading preprocessing pipeline...")

    try:
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(
                f"Preprocessor not found: {PREPROCESSOR_PATH}"
            )

        preprocessor = joblib.load(PREPROCESSOR_PATH)

        # Use the exact input columns recorded during training.
        if not hasattr(preprocessor, "feature_names_in_"):
            raise RuntimeError(
                "Preprocessor has no feature_names_in_. "
                "Check the saved preprocessing pipeline."
            )

        EXPECTED_COLUMNS = list(preprocessor.feature_names_in_)

        logger.info("Preprocessor loaded successfully.")
        logger.info("Expected columns: %s", EXPECTED_COLUMNS)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        logger.info("Loading champion model from MLflow...")

        model = mlflow.sklearn.load_model(MODEL_URI)

        logger.info("Champion model loaded successfully.")

    except Exception:
        logger.exception("Application startup failed.")
        raise

    yield


# --------------------------------------------------
# 6. Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Healthcare Fraud Detection API",
    description="Predict potential healthcare claim fraud.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# 7. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Healthcare Fraud Detection API is running.",
        "docs": "/docs",
        "health": "/health",
    }


# --------------------------------------------------
# 8. Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "preprocessor_loaded": preprocessor is not None,
        "model_loaded": model is not None,
    }


# --------------------------------------------------
# 9. Prepare incoming claim data
# --------------------------------------------------

def transform_claim(claim: ClaimInput):
    if preprocessor is None or EXPECTED_COLUMNS is None:
        raise RuntimeError("Preprocessor is not loaded.")

    # Convert Pydantic input into a DataFrame.
    data = pd.DataFrame([claim.model_dump()])

    # Parse the submission date.
    date_column = "Claim_Submission_Date"

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce",
    )

    if data[date_column].isna().any():
        raise ValueError(
            "Invalid Claim_Submission_Date. "
            "Use a valid date such as 2026-10-20."
        )

    # IMPORTANT:
    # These names must match the training pipeline exactly.
    data["Claim_Submission_Date_month"] = (
        data[date_column].dt.month
    )

    data["Claim_Submission_Date_day_of_week"] = (
        data[date_column].dt.dayofweek
    )

    # Keep only the columns the preprocessor expects.
    # This removes the original date column if it was
    # replaced by the two engineered date features.
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

    data = data[EXPECTED_COLUMNS]

    # Apply the same fitted transformations used in training.
    transformed = preprocessor.transform(data)

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
            probability = float(probabilities[0][1])

        # Step 4: Return a structured response.
        return {
            "prediction": prediction,
            "is_fraud": prediction,
            "fraud_probability": probability,
            "result": (
                "Potential fraud detected"
                if prediction == 1
                else "No fraud detected by the model"
            ),
            "message": (
                "This is a model prediction, not a confirmed "
                "finding of fraud."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Prediction failed.")

        # Show a useful error in the API response while
        # preserving the full traceback in the terminal.
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc
