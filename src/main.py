import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import sys

# Adds the project root to the path so it can see the 'src' folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_model import inference  # noqa: E402
from src.data import process_data   # noqa: E402

app = FastAPI()


current_dir = os.path.dirname(os.path.abspath(__file__))
# Load artifacts on startup
# Go UP one level to the root, then INTO the 'model' folder
model_path = os.path.join(current_dir, "..", "model", "trained_model.pkl")
encoder_path = os.path.join(current_dir, "..", "model", "encoder.pkl")
lb_path = os.path.join(current_dir, "..", "model", "label_binarizer.pkl")

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)
lb = joblib.load(lb_path)


class CensusData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlwgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status",
                                example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country",
                                example="United-States")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlwgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        },
    }


@app.get("/")
async def root():
    return {"message": "Welcome to the Census Income Prediction API!"}


@app.post("/predict")
async def predict(data: CensusData):
    # Convert Pydantic model to DataFrame
    input_df = pd.DataFrame([data.model_dump(by_alias=True)])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Process the data (training=False because we are using saved artifacts)
    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    prediction = inference(model, X)
    label = lb.inverse_transform(prediction)[0]

    return {"prediction": label}
