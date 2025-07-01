import sklearn
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imblearn_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OrdinalEncoder
def get_input():    
    male = st.selectbox("select gender [1 for male o for female]",options=[1,0])
    age=st.slider("select Age" , min_value=32 , max_value= 70 , value=40)
    education=st.selectbox("select education level [1 for male o for female]",options=[1,2,3,4])
    currentSmoker=st.selectbox("if you currently smoke chosse 1",options=[0,1])
    cigsPerDay=st.slider("select number of cigarettes per day if you smoke if not select 0  " , min_value=0 , max_value= 60 , value=20)
    BPMeds=st.selectbox("if you take BPMeds choose 1",options=[0,1])
    prevalentStroke=st.selectbox("if you had prevalentStroke choose 1",options=[0,1])
    prevalentHyp=st.selectbox("if you had prevalentHyp choose 1",options=[0,1])
    diabetes=st.selectbox("if you have diabetes choose 1",options=[0,1])
    totChol=st.slider("select totChol" , min_value=107 , max_value= 600 , value=300)
    sysBP = st.slider("select Systolic Blood Pressure", min_value=83.5, max_value=295.0, value=100.0)
    diaBP = st.slider("select Diastolic Blood Pressure", min_value=48.0, max_value=142.5, value=100.0)
    BMI = st.slider("select Body Mass Index", min_value=15.54, max_value=45.8, value=30.0)
    heartRate=st.slider("select heartRate pressure" , min_value=44 , max_value=115  , value=60)
    glucose=st.slider("select heartRate pressure" , min_value=40 , max_value=394  , value=200)
    BP_Cat=st.selectbox("select Blood Pressure ",options=['Normal', 'High (Hypertension Stage 1)','High (Hypertension Stage 2)', 'Elevated',
           'Hypertensive Crisis (Seek Medical Attention)',
           'Low (Hypotension)'])
    BMI_category=st.selectbox("select BMI_category",options=['Overweight', 'Normal Weight', 'Obese (Class 1)',
           'Obese (Class 2)', 'Obese (Class 3)', 'Underweight'])
    heartRate_cat=st.selectbox("select heartRate_cat",options=['Normal', 'Bradycardia (Low)', 'Tachycardia (High)'])
    glucose_cat=st.selectbox("select glucose_cat",options=['Normal', 'Prediabetes', 'Diabetes'])
    totChol_cat=st.selectbox("select totChol_cat",options=['Desirable', 'High', 'BorderLine'])
    return pd.DataFrame(    [[male,
                              age,
                              education,
                              currentSmoker,
                              cigsPerDay,
                              BPMeds,
                              prevalentStroke,
                              prevalentHyp,
                              diabetes,
                              totChol,
                              sysBP,
                              diaBP,
                              BMI,
                              heartRate,
                             glucose,
                              BP_Cat,
                              BMI_category,
                              heartRate_cat,
                              glucose_cat,
                              totChol_cat]],columns=joblib.load("features.h5"))
test=get_input()
st.dataframe(test)
model = joblib.load("best_logreg.h5")
st.write("You may be at risk of heart disease. Please undergo the necessary medical check-ups."if model.predict(test)==1 else 
"You are currently not at risk of developing heart disease. No further medical examinations are required at this time.")
