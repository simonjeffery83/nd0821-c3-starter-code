import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
import os, sys

# Adds the project root to the path so it can see the 'src' folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import process_data


# --- CORE FUNCTION 1: TRAINING ---
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.
    """
    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    model.fit(X_train, y_train)
    return model


# --- CORE FUNCTION 2: METRICS ---
def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


# --- CORE FUNCTION 3: INFERENCE ---
def inference(model, X):
    """
    Run machine learning predictions and return the predictions.
    """
    preds = model.predict(X)
    return preds


# --- SLICING PERFORMANCE FUNCTION ---
def compute_slices(df, feature, model, encoder, lb, cat_features):
    """
    Outputs the performance of the model on slices of a categorical feature.
    """
    results = []
    for value in df[feature].unique():
        slice_df = df[df[feature] == value]

        x_slice, y_slice, _, _ = process_data(
            slice_df,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )

        preds = inference(model, x_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)

        line = f"Feature: {feature} | Value: {value} | Precision: {precision:.2f} | Recall: {recall:.2f}"
        print(line)
        results.append(line)

    with open("slice_output.txt", "a") as f:
        f.write(f"--- Slices for {feature} ---\n")
        for res in results:
            f.write(res + "\n")


# --- MAIN EXECUTION SCRIPT ---
if __name__ == "__main__":
    # Load data

    data_path = "data/adult.data"

    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "salary",
    ]

    data = pd.read_csv(data_path, names=columns, sep=",", skipinitialspace=True)
    train, test = train_test_split(data, test_size=0.20, random_state=42)

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

    # Process training data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # 1. Train Model
    model = train_model(X_train, y_train)

    # 2. Process Test Data & Run Inference
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X_test)

    # 3. Compute overall metrics
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(
        f"Overall Metrics -> Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {fbeta:.2f}"
    )

    # 4. Save artifacts
    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(model, os.path.join(model_dir, "trained_model.pkl"))
    joblib.dump(encoder, os.path.join(model_dir, "encoder.pkl"))
    joblib.dump(lb, os.path.join(model_dir, "label_binarizer.pkl"))

    # 5. Run Slicing (example on 'education')
    compute_slices(test, "education", model, encoder, lb, cat_features)
