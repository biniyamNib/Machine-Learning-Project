# Import necessary libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import asyncio
from typing import Dict, Any

# Load the trained model
def load_model(model_path: str):
    """Load the trained model from a file."""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load the model: {e}")

# Load the model (replace with the path to your saved model)
model = load_model("rain_prediction_model.joblib")

# Define the input data schema using Pydantic
class InputData(BaseModel):
    MinTemp: float
    MaxTemp: float
    Rainfall: float
    Evaporation: float
    Sunshine: float
    WindGustDir: str
    WindGustSpeed: float
    WindDir9am: str
    WindDir3pm: str
    WindSpeed9am: float
    WindSpeed3pm: float
    Humidity9am: float
    Humidity3pm: float
    Pressure9am: float
    Pressure3pm: float
    Cloud9am: float
    Cloud3pm: float
    Temp9am: float
    Temp3pm: float
    RainToday: str

# Initialize FastAPI app
app = FastAPI()

# Example asynchronous function
async def fetch_data() -> str:
    """Simulate an asynchronous I/O operation."""
    await asyncio.sleep(1)
    return "Data fetched"

@app.get("/")
async def read_root() -> Dict[str, str]:
    """Root endpoint that demonstrates asynchronous functionality."""
    result = await fetch_data()
    return {"message": result}

# Define the prediction endpoint
@app.post("/predict")
async def predict(data: InputData) -> Dict[str, Any]:
    """Endpoint to make predictions using the trained model."""
    try:
        # Convert input data to a DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Run the model prediction in a separate thread to avoid blocking the event loop
        prediction, prediction_proba = await asyncio.to_thread(
            lambda: (
                model.predict(input_data),
                model.predict_proba(input_data)[:, 1],
            )
        )

        # Return the prediction
        return {
            "prediction": "Yes" if prediction[0] == 1 else "No",
            "probability": float(prediction_proba[0]),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

# Save the trained model to a file (run this once before deploying)
# joblib.dump(best_model, "rainfall_model.pkl")