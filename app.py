import streamlit as st
import pandas as pd
import joblib
from PIL import Image

st.set_page_config(page_title="Mental Health Predictor", page_icon="🧠")
st.title("🧠 Mental Health Support Predictor")
st.write("_Fill the confidential form below to assess your mental health support need._")

# Load model and features
model = joblib.load('mental_health_model.pkl')
feature_names = joblib.load('feature_names.pkl')

# Styled form layout
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("🎂 Age", 18, 65, 30)
        gender = st.selectbox("⚧️ Gender", ["Male", "Female", "Other"])
        self_employed = st.selectbox("💼 Self-employed?", ["Yes", "No"])
        family_history = st.selectbox("👨‍👩‍👧 Family history of mental illness?", ["Yes", "No"])
        work_interfere = st.selectbox("📉 Work interference due to mental health", ["Never", "Rarely", "Sometimes", "Often", "Don’t know"])
        no_employees = st.selectbox("🏢 Company size", ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000'])
        remote_work = st.selectbox("🏠 Work remotely?", ["Yes", "No"])
        tech_company = st.selectbox("💻 Tech company?", ["Yes", "No"])
    
    with col2:
        benefits = st.selectbox("🧾 Mental health benefits offered?", ["Yes", "No", "Don't know"])
        care_options = st.selectbox("🩺 Aware of care options?", ["Yes", "No", "Not sure"])
        wellness_program = st.selectbox("🏋️‍♂️ Wellness programs available?", ["Yes", "No", "Don't know"])
        seek_help = st.selectbox("🙋 Encouraged to seek help?", ["Yes", "No", "Don't know"])
        anonymity = st.selectbox("🕶️ Anonymity guaranteed?", ["Yes", "No", "Don't know"])
        leave = st.selectbox("📅 Ease of taking leave", ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"])
        mental_health_consequence = st.selectbox("🧠 Consequences for mental health disclosure?", ["Yes", "No", "Maybe"])
        phys_health_consequence = st.selectbox("💪 Consequences for physical health disclosure?", ["Yes", "No", "Maybe"])
        coworkers = st.selectbox("👥 Talk to coworkers?", ["Yes", "No", "Some of them"])
        supervisor = st.selectbox("👨‍💼 Talk to supervisor?", ["Yes", "No", "Some of them"])

    col3, col4 = st.columns(2)
    with col3:
        mental_health_interview = st.selectbox("🧠 Discuss mental health in interview?", ["Yes", "No", "Maybe"])
        phys_health_interview = st.selectbox("💪 Discuss physical health in interview?", ["Yes", "No", "Maybe"])
    with col4:
        mental_vs_physical = st.selectbox("⚖️ Importance of mental vs physical health", ["Yes", "No", "Don't know"])
        obs_consequence = st.selectbox("👀 Seen mental health consequences at work?", ["Yes", "No"])
        country = st.selectbox("🌍 Country", ["United States", "India", "Canada", "United Kingdom", "Germany", "Others"])

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    input_dict = {
        'Age': age,
        'Gender': gender,
        'self_employed': self_employed,
        'family_history': family_history,
        'work_interfere': work_interfere,
        'no_employees': no_employees,
        'remote_work': remote_work,
        'tech_company': tech_company,
        'benefits': benefits,
        'care_options': care_options,
        'wellness_program': wellness_program,
        'seek_help': seek_help,
        'anonymity': anonymity,
        'leave': leave,
        'mental_health_consequence': mental_health_consequence,
        'phys_health_consequence': phys_health_consequence,
        'coworkers': coworkers,
        'supervisor': supervisor,
        'mental_health_interview': mental_health_interview,
        'phys_health_interview': phys_health_interview,
        'mental_vs_physical': mental_vs_physical,
        'obs_consequence': obs_consequence,
        'Country': country
    }

    input_df = pd.DataFrame([input_dict])
    input_df_encoded = pd.get_dummies(input_df)

    for col in feature_names:
        if col not in input_df_encoded:
            input_df_encoded[col] = 0
    input_df_encoded = input_df_encoded[feature_names]

    prediction = model.predict(input_df_encoded)[0]
    result = "🟢 No Immediate Support Needed" if prediction == 0 else "🔴 Likely Needs Mental Health Support"
    st.markdown(f"## Prediction: {result}")
