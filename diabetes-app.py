import streamlit as st
import pickle
import numpy as np 
with open('GROUP_05.pkl','rb') as file:
    load_model=pickle.load(file)

def deb_prediction(test_data):
    Array_test_data = np.asarray(test_data)
    reshaped_array = Array_test_data.reshape(1,-1)
    prediction = load_model.predict(reshaped_array)
    return (prediction)
def main():
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
    
    diagnosis = ''
    
    
    if st.button("predict patient diabetes"):
        diagnosis = deb_prediction([PG,GS,BP,ST,IS,BM,DP,AG])
    st.success(diagnosis)

if __name__ == '__main__':
    main()
