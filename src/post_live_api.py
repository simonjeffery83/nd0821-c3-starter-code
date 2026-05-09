import requests

# Replace with your actual Heroku app URL
live_url =\
    "https://fast-api-demo-eda961a644d7.herokuapp.com/predict"


data = {
    "age": 32,
    "workclass": "Private",
    "fnlwgt": 205019,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 45,
    "native-country": "United-States",
}

response = requests.post(live_url, json=data)

print(f"Response Status Code: {response.status_code}")
print(f"Model Prediction: {response.json()['prediction']}")
