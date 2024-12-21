import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import joblib

# building the interface
st.title('Medical Costs Prediction App')
st.image('health-costs.jpg')
st.text('This application can predict the costs for your next medical treatment')
st.text('Please fill-in the following data:')

age = st.slider("Age", 18, 65)
bmi = st.number_input("BMI")
child = st.number_input("No. of Children")
sex = st.selectbox("Gender", ['Male','Female'])
smoking = st.selectbox("Somker", ['Yes','No'])
region = st.radio("Region", ['North East', 'North West', 'South East', 'South West'])
btn = st.button("Predict")

if btn == True:
    scaler = joblib.load('scaler.pkl')
    target_scaler = joblib.load('target_scaler.pkl')
    model = joblib.load('model.pkl')

    # encoding categorical data
    sex_map = {'Female':0, 'Male':1}
    smoking_map = {'Yes':1, 'No':0}
    region_map = {'North East':0, 'North West':1, 'South East':2, 'South West':3}

    sex_encoded = sex_map[sex]
    smoking_encoded = smoking_map[smoking]
    region_encoded = region_map[region]

    input_data = np.array([[age,bmi,child,sex_encoded,smoking_encoded,region_encoded]])
    
    input_data_scaled = scaler.transform(input_data)

    pred_scaled = model.predict(input_data_scaled)

    pred_org = target_scaler.inverse_transform(pred_scaled.reshape(-1,1))
    st.success("Your cost will be "+ str(pred_org))