import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

# Load model pipeline
model = joblib.load('stacked_pipeline1.joblib')

# Input features using sliders
inputs = {}
for i in range(1, 29):
    inputs[f'V{i}'] = st.slider(f'PCA Component V{i}', min_value=-50.0, max_value=50.0, value=0.0, step=0.1)

# Use sliders for Time and Amount
time_input = st.slider('Time (seconds)', min_value=0.0, max_value=200000.0, value=0.0, step=1.0)
amount_input = st.slider('Amount', min_value=0.0, max_value=2500.0, value=0.0, step=1.0)

if st.button('Predict Fraud'):
    # Scale Time and Amount
    #time_scaled, amount_scaled = scaler.transform([[time_input, amount_input]])[0]
    inputs['Time'] = time_input
    inputs['Amount'] = amount_input

    # Convert inputs to array
    input_array = np.array([list(inputs.values())])
    prediction = model.predict(input_array)
    proba = model.predict_proba(input_array)[0][1]

    st.write(f'Fraud Prediction: {"Yes" if prediction[0]==1 else "No"}')
    st.write(f'Fraud Probability: {proba:.2f}')
