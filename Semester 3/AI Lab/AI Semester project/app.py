import streamlit as st
import joblib
import pandas as pd

model = joblib.load(r'D:\Downloads\crop_yield_model.pkl')

st.title('Crop Yield Prediction')
st.sidebar.header('Input Parameters')

region = st.sidebar.selectbox('Region', ['North', 'South', 'East', 'West'])
soil_type = st.sidebar.selectbox('Soil Type', ['Loamy', 'Sandy', 'Clay', 'Saline']) 
crop = st.sidebar.selectbox('Crop', ['Wheat', 'Rice', 'Maize'])
rainfall = st.sidebar.number_input('Rainfall (mm)', min_value=0, max_value=500, value=100)
temperature = st.sidebar.number_input('Temperature (Celsius)', min_value=-10, max_value=50, value=25)
fertilizer_used = st.sidebar.selectbox('Fertilizer Used', ['Yes', 'No'])
irrigation_used = st.sidebar.selectbox('Irrigation Used', ['Yes', 'No'])
weather_condition = st.sidebar.selectbox('Weather Condition', ['Sunny', 'Cloudy', 'Rainy'])
days_to_harvest = st.sidebar.number_input('Days to Harvest', min_value=30, max_value=365, value=90)

region_mapping = {'North': 0, 'South': 1, 'East': 2, 'West': 3}
soil_type_mapping = {'Loamy': 0, 'Sandy': 1, 'Clay': 2, 'Saline': 3}
crop_mapping = {'Wheat': 0, 'Rice': 1, 'Maize': 2}
fertilizer_mapping = {'Yes': 1, 'No': 0}
irrigation_mapping = {'Yes': 1, 'No': 0}
weather_condition_mapping = {'Sunny': 0, 'Cloudy': 1, 'Rainy': 2}

region_numeric = region_mapping.get(region, 0)
soil_type_numeric = soil_type_mapping.get(soil_type, 0)
crop_numeric = crop_mapping.get(crop, 0)
fertilizer_numeric = fertilizer_mapping.get(fertilizer_used, 0)
irrigation_numeric = irrigation_mapping.get(irrigation_used, 0)
weather_condition_numeric = weather_condition_mapping.get(weather_condition, 0)

input_data = pd.DataFrame({
    'Region': [region_numeric],
    'Soil_Type': [soil_type_numeric],
    'Crop': [crop_numeric],
    'Rainfall_mm': [rainfall],
    'Temperature_Celsius': [temperature],
    'Fertilizer_Used': [fertilizer_numeric],
    'Irrigation_Used': [irrigation_numeric],
    'Weather_Condition': [weather_condition_numeric],
    'Days_to_Harvest': [days_to_harvest]
})


prediction = model.predict(input_data)
st.write(f"Predicted Crop Yield: {prediction[0]} units")
