import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Loading the save saved pipeline
pipeline = joblib.load('final_random_forest_pipeline.pkl')

# Specifying Numercial and Binary features for appropriate data hadnling
num_features = [
    'hemoglobin', 'serum_creatinine', 'blood_glucose_random',
    'albumin', 'red_blood_cell_count', 'potassium', 'blood_urea'
]
binary_features = ['appetite', 'hypertension', 'pedal_edema']

st.title("Cancer Disease Prediction")

st.markdown("Enter patient data below to get prediction results.")

# Collecting numerical inputs
numerical_input = []
for feature in num_features:
    val = st.number_input(f"{feature.replace('_', ' ').capitalize()}", value=0.0)
    numerical_input.append(val)

# Collecting binary inputs
binary_input = []
for feature in binary_features:
    val = st.selectbox(f"{feature.replace('_', ' ').capitalize()} (0=No, 1=Yes)", options=[0, 1])
    binary_input.append(val)

if st.button("Predict"):
    # Combining inputs in correct feature order
    full_input_list = numerical_input + binary_input

    # Converting to DataFrame with correct columns
    input_df = pd.DataFrame([full_input_list], columns=num_features + binary_features)

    # Passing full input through the preprocessing pipeline
    preprocessed_input = pipeline.named_steps['preprocess'].transform(input_df)

    # Predicting using classifier step
    clf = pipeline.named_steps['clf']
    prediction = clf.predict(preprocessed_input)[0]
    proba = clf.predict_proba(preprocessed_input)[0][1]

    # Displaying results
    st.subheader("Prediction Result")
    st.write(f"Predicted Class: {'Disease Detected' if prediction == 1 else 'No Disease'}")
    st.write(f"Probability of Disease: {proba:.4f}")
