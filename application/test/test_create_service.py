import requests
import json

def test_service():
    prediction = requests.post(
        "http://127.0.0.1:3000/predict",
        headers={"content-type": "application/json"},
        data=json.dumps({
        "employee": {
            "Education": "Bachelors",
            "JoiningYear": 2017,
            "City": "Pune",
            "PaymentTier": 1,
            "Age": 25,
            "Gender": "Female",
            "EverBenched": "No",
            "ExperienceInCurrentDomain": 1
        }
        })
    ).text
    assert prediction[0] in ["0", "1"]


