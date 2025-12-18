import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Configuration & Loading ---
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

@st.cache_resource
def load_assets():
    """Load the model and scaler once and cache them."""
    try:
        # Loading the joblib versions as they are generally more stable across versions
        model = joblib.load("GROUP_05_model.joblib")
        scaler = joblib.load("GROUP_05_scaler.joblib")
        return model, scaler
    except FileNotFoundError:
        st.error("Model or Scaler files not found. Please run the training script first.")
        return None, None

model, scaler = load_assets()

# --- 2. User Interface ---
st.title("🩺 Diabetes Risk Predictor")
st.markdown("Enter the patient's clinical data below to predict the likelihood of diabetes.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
        bp = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
        skin = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
        
    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI (Weight in kg/(m)^2)", min_value=0.0, max_value=70.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
        age = st.number_input("Age (Years)", min_value=0, max_value=120, value=30)

    submit = st.form_submit_button("Predict Result")

# --- 3. Prediction Logic ---
if submit:
    if model is not None and scaler is not None:
        # Create a dataframe for the input
        input_data = pd.DataFrame([[
            pregnancies, glucose, bp, skin, insulin, bmi, dpf, age
        ]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        
        # Preprocessing: Impute zeros with medians (mimicking the training logic)
        # Note: In a production app, you'd use the training medians here.
        
        # Scale the data
        scaled_input = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]
        
        # Display Results
        st.divider()
        if prediction == 1:
            st.error(f"### Result: High Risk (Positive)")
            st.write(f"Confidence Level: **{probability:.2%}**")
        else:
            st.success(f"### Result: Low Risk (Negative)")
            st.write(f"Confidence Level: **{(1-probability):.2%}**")
            
        # Metric display
        st.metric("Probability of Diabetes", f"{probability:.1%}")
    else:
        st.warning("Prediction unavailable: Model assets missing.")

st.info("Disclaimer: This is a machine learning tool and should not replace professional medical advice.")
