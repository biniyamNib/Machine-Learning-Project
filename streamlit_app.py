import streamlit as st
import requests
import pandas as pd

# FastAPI endpoint
FASTAPI_URL = "http://127.0.0.1:8000/predict"

# Title of the app
st.title("🌧️ Rain Prediction App")

# Description
st.write("""
This app predicts whether it will rain tomorrow based on weather data. 
Enter the required information below and click **Predict** to see the result.
""")

# Input fields for the user
st.header("Input Features")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    MinTemp = st.number_input("Minimum Temperature (°C)", value=10.0)
    MaxTemp = st.number_input("Maximum Temperature (°C)", value=20.0)
    Rainfall = st.number_input("Rainfall (mm)", value=0.0)
    Evaporation = st.number_input("Evaporation (mm)", value=5.0)
    Sunshine = st.number_input("Sunshine (hours)", value=7.0)
    WindGustDir = st.selectbox("Wind Gust Direction", ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    WindGustSpeed = st.number_input("Wind Gust Speed (km/h)", value=40.0)
    WindDir9am = st.selectbox("Wind Direction at 9am", ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    WindDir3pm = st.selectbox("Wind Direction at 3pm", ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

with col2:
    WindSpeed9am = st.number_input("Wind Speed at 9am (km/h)", value=10.0)
    WindSpeed3pm = st.number_input("Wind Speed at 3pm (km/h)", value=20.0)
    Humidity9am = st.number_input("Humidity at 9am (%)", value=70.0)
    Humidity3pm = st.number_input("Humidity at 3pm (%)", value=50.0)
    Pressure9am = st.number_input("Pressure at 9am (hPa)", value=1015.0)
    Pressure3pm = st.number_input("Pressure at 3pm (hPa)", value=1010.0)
    Cloud9am = st.number_input("Cloud at 9am (oktas)", value=5.0)
    Cloud3pm = st.number_input("Cloud at 3pm (oktas)", value=4.0)
    Temp9am = st.number_input("Temperature at 9am (°C)", value=15.0)
    Temp3pm = st.number_input("Temperature at 3pm (°C)", value=18.0)
    RainToday = st.selectbox("Rain Today", ["No", "Yes"])

# Predict button
if st.button("Predict"):
    # Prepare the input data
    input_data = {
        "MinTemp": MinTemp,
        "MaxTemp": MaxTemp,
        "Rainfall": Rainfall,
        "Evaporation": Evaporation,
        "Sunshine": Sunshine,
        "WindGustDir": WindGustDir,
        "WindGustSpeed": WindGustSpeed,
        "WindDir9am": WindDir9am,
        "WindDir3pm": WindDir3pm,
        "WindSpeed9am": WindSpeed9am,
        "WindSpeed3pm": WindSpeed3pm,
        "Humidity9am": Humidity9am,
        "Humidity3pm": Humidity3pm,
        "Pressure9am": Pressure9am,
        "Pressure3pm": Pressure3pm,
        "Cloud9am": Cloud9am,
        "Cloud3pm": Cloud3pm,
        "Temp9am": Temp9am,
        "Temp3pm": Temp3pm,
        "RainToday": RainToday,
    }

    # Send the input data to the FastAPI endpoint
    response = requests.post(FASTAPI_URL, json=input_data)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['prediction']}")
        st.info(f"Probability: {result['probability']:.2f}")
    else:
        st.error("An error occurred while making the prediction. Please check your input.")