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
        with st.status("In Porgess, Wait a minute...", expanded=True) as status:

            st.write("Send Data to model...")
            payload = {"employee": data}
            data_json = json.dumps(payload)

            st.write("Model Predict...")
            try:
                prediction = int(requests.post(
                        "https://employee-churn-1-623372933969.asia-east1.run.app/predict",
                        headers={"content-type": "application/json"},
                        data=data_json,
                    ).text[0])
                st.write("Return Answer...")
                status.update(
                    label="Predict complete!", state="complete", expanded=False
                )
            except requests.exceptions.RequestException as e:
                st.write(f"Error: {e}")
                
        if prediction == 0:
            st.success("This employee is predicted to stay more than two years.")
        else:
            st.warning("This employee is predicted to leave in two years.")




def main():
    data = data_input()
    write_predictions(data)


if __name__ == "__main__":
    main()
