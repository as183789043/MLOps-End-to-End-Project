import json
import requests
import streamlit as st
import datetime

# Current Year
currentDateTime = datetime.datetime.now()
date = currentDateTime.date()
year = date.strftime("%Y")


def data_input():
    st.title("Predict Employee Future")
    data = {}

    data['Education'] = st.selectbox(
        "Choose Education",
        options=['Bachelors', 'Masters', 'PHD'],
    )

    data['JoiningYear'] = st.number_input(
        'Choose Joining Year', min_value=1990, max_value=int(year), step=1)

    data['City'] = st.selectbox("Choose City",
                                options=["Bangalore", "New Delhi", "Pune"])

    data['PaymentTier'] = st.number_input('Choose Payment Tier', min_value=1, max_value=3, step=1)

    data['Age'] = st.number_input(
        'Choose Age', min_value=18, max_value=65, step=1)

    data['Gender'] = st.selectbox('Choose Gender', options=['Male', 'Female'])

    data['EverBenched'] = st.selectbox(
        "Ever kept out of projects for 1 month or more",
        options=["Yes", "No"]
    )

    data['ExperienceInCurrentDomain'] = st.number_input(
        'Experience In Current Field', min_value=0, step=1, value=1)

    return data


def write_predictions(data: dict):
    if st.button("Will this employee leave in 2 years?"):
        payload = {"employee": data}
        data_json = json.dumps(payload)

        try:
            prediction = int(requests.post(
                "http://localhost:3000/predict",
                headers={"content-type": "application/json"},
                data=data_json,
            ).text[0])
            

            if prediction == 0:
                st.write("This employee is predicted to stay more than two years.")
            else:
                st.write("This employee is predicted to leave in two years.")

        except requests.exceptions.RequestException as e:
            st.write(f"Error: {e}")


def main():
    data = data_input()
    write_predictions(data)


if __name__ == "__main__":
    main()
