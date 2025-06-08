import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model, encoder, and feature names
model = joblib.load('mental_health_model.pkl')
encoder = joblib.load('encoder.pkl')  # OneHotEncoder for work_interfere
feature_names = joblib.load('feature_names.pkl')

st.title("Mental Health Predictor")
st.write("Fill the form to predict your mental health support need.")

# Input fields (add any other features your model uses!)
age = st.slider('Age', 18, 65, 30)
gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
family_history = st.selectbox('Family History of Mental Illness?', ['Yes', 'No'])
work_interfere = st.selectbox('Work Interference', ['Never', 'Rarely', 'Sometimes', 'Often', "Don’t know"])

# Step 1: Prepare input dataframe
input_df = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'family_history': [family_history],
    'work_interfere': [work_interfere],
    # Add other inputs as needed
})

# Step 2: Label encode binary columns with the same mapping as training

gender_map = {'Male': 1, 'Female': 0, 'Other': 2}  # as per LabelEncoder results, double-check your mappings!
family_history_map = {'Yes': 1, 'No': 0}

input_df['Gender'] = input_df['Gender'].map(gender_map)
input_df['family_history'] = input_df['family_history'].map(family_history_map)

# Step 3: One-hot encode work_interfere using loaded encoder
# The encoder expects 2D array input, so reshape appropriately
work_interfere_encoded = encoder.transform(input_df[['work_interfere']])
work_interfere_df = pd.DataFrame(
    work_interfere_encoded, 
    columns=[f'work_interfere_{cat}' for cat in encoder.categories_[0]]
)

# Drop original 'work_interfere' and concat encoded
input_df = input_df.drop('work_interfere', axis=1)
input_df = pd.concat([input_df.reset_index(drop=True), work_interfere_df], axis=1)

# Step 4: Add any missing columns your model expects but input does not have
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0  # Fill missing columns with 0

# Step 5: Reorder columns to match training
input_df = input_df[feature_names]

# Step 6: Predict
if st.button('Predict'):
    prediction = model.predict(input_df)[0]
    st.success(f"Prediction: {'Needs Support' if prediction == 1 else 'No Immediate Need'}")
