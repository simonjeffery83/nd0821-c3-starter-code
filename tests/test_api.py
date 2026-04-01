from fastapi.testclient import TestClient
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.main import app

client = TestClient(app)

def test_get_root():
    
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Census Income Prediction API!"}

def test_post_predict_low_income():
    # Example that typically leads to <=50K
    data = {
        "age": 20, "workclass": "Private", "fnlwgt": 1000, "education": "HS-grad",
        "education-num": 9, "marital-status": "Never-married", "occupation": "Other-service",
        "relationship": "Own-child", "race": "Black", "sex": "Male",
        "capital-gain": 0, "capital-loss": 0, "hours-per-week": 20, "native-country": "United-States"
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"

def test_post_predict_high_income():
    # Example that typically leads to >50K
    data = {
        "age": 50, "workclass": "Private", "fnlwgt": 200000, "education": "Masters",
        "education-num": 14, "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial",
        "relationship": "Husband", "race": "White", "sex": "Male",
        "capital-gain": 15000, "capital-loss": 0, "hours-per-week": 60, "native-country": "United-States"
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"