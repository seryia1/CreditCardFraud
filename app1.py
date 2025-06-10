import streamlit as st
import numpy as np
import joblib

# Load model pipeline
model = joblib.load('stacked_pipeline.joblib')

# Input features
inputs = {}
for i in range(1, 29):
    inputs[f'V{i}'] = st.number_input(f'PCA Component V{i}', value=0.0)
inputs['Time'] = st.number_input('Time (seconds)', value=0.0)
inputs['Amount'] = st.number_input('Amount', value=0.0)

if st.button('Predict Fraud'):
    # Convert inputs to array
    input_array = np.array([list(inputs.values())])
    prediction = model.predict(input_array)
    proba = model.predict_proba(input_array)[0][1]
    st.write(f'Fraud Prediction: {"Yes" if prediction[0]==1 else "No"}')
    st.write(f'Fraud Probability: {proba:.2f}')
