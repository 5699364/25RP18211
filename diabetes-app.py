import streamlit as st
import pickle
import numpy as np 
with open('GROUP_05.pkl','rb') as file:
    best_gb_model=pickle.load(file)
    
st.title("DIABETES PREDICTIONS")
st.write("Enter The Following Data For Predictions")
PG=st.number_input("Pregnancies",value=0)
GS=st.number_input("Glucose",value=0)
BP=st.number_input("BloodPressure",value=0)
ST=st.number_input("Skin Thickness",value=0)
IS=st.number_input("Insulin",value=0)
BM=st.number_input("BMI",value=0)
DP=st.number_input("Diabetes Pedigree Function",value=0)
AG=st.number_input("Age",value=0)
if st.button("predict patient diabetes"):
    Input=np.array([[PG,GS,BP,ST,IS,BM,DP,AG]])
    prediction=best_gb_model.predict(Input)
    st.success(prediction[0])

