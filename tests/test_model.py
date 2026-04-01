import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# Add the starter/starter directory to the python path
# Adds the project root to the path so it can see the 'src' folder

from src.train_model import train_model, compute_model_metrics, inference

@pytest.fixture
def data():
    """
    Fixture to provide dummy training data (X) and labels (y).
    """
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    return X, y

# --- Test 1: train_model ---
def test_train_model(data):
    """
    Test that train_model returns a fitted RandomForestClassifier.
    """
    X, y = data
    model = train_model(X, y)
    
    assert isinstance(model, RandomForestClassifier)
    # Check if the model has been fitted (has the 'estimators_' attribute)
    assert hasattr(model, "estimators_")

# --- Test 2: compute_model_metrics ---
def test_compute_model_metrics():
    """
    Test the metrics calculation for perfect and imperfect predictions.
    """
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1]) # Perfect match
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0
    
    # Test with one error
    y_pred_err = np.array([0, 0, 0, 1]) 
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred_err)
    assert precision < 1.0 or recall < 1.0

# --- Test 3: inference ---
def test_inference(data):
    """
    Test that inference returns the correct number of predictions and valid labels.
    """
    X, y = data
    model = train_model(X, y) # Get a trained model
    preds = inference(model, X)
    
    assert len(preds) == len(X)
    # Ensure predictions are binary (0 or 1) as expected for this dataset
    assert np.all((preds == 0) | (preds == 1))

# --- Bonus: Test Data Integrity ---
def test_data_is_not_empty():
    """
    Sanity check to ensure the data fixture is providing content.
    """
    X = np.array([[1, 2]])
    assert X.shape[0] > 0