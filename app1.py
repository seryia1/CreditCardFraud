import streamlit as st
import numpy as np
import joblib

# Load your stacked model
model = joblib.load('stacked_model.pkl')

# Create input fields for PCA components, Time and Amount
inputs = {}
for i in range(1, 29):
    inputs[f'V{i}'] = st.number_input(f'PCA Component V{i}', value=0.0)

inputs['Time'] = st.number_input('Time (seconds)', value=0.0)
inputs['Amount'] = st.number_input('Amount', value=0.0)

if st.button('Predict Fraud'):
    # Prepare input array for prediction
    input_array = np.array([list(inputs.values())])
    prediction = model.predict(input_array)
    proba = model.predict_proba(input_array)[0][1]
    st.write(f'Fraud Prediction: {"Yes" if prediction[0]==1 else "No"}')
    st.write(f'Fraud Probability: {proba:.2f}')
