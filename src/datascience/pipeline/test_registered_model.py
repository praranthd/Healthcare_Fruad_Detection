import mlflow
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.sklearn.load_model(
    "models:/HealthcareFraudDetectionModel@champion"
)

X_test = pd.read_csv(
    "artifacts/data_transformation/X_test.csv"
)

sample = X_test.iloc[[0]]

prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]

fraud_index = list(model.classes_).index(1)

print("Prediction:", int(prediction))
print(f"Fraud probability: {probabilities[fraud_index]:.4f}")