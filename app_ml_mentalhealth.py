import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pycountry
from tensorflow.keras.models import load_model


# --- Load Model and Column Names ---
model = load_model("mental_health_nn_model.h5")
model_columns = joblib.load("model_columns.pkl")

# Load one-hot encoded column template (should be one row with all 112 cols)
template = pd.read_csv("survey.csv")
template = template.iloc[0:1]

st.title("🧠 Mental Health Risk Predictor")

# --- USER INPUTS ---
age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
all_countries = sorted([country.name for country in pycountry.countries])
country = st.selectbox("Country", all_countries)
state = st.text_input("State (if US)", "California")
self_employed = st.selectbox("Are you self-employed?", ["Select", "College Student", "Yes", "No", "Don't know"])
family_history = st.selectbox("Family history of mental illness?", ["Yes", "No"])
treatment = st.selectbox("Have you sought mental health treatment?", ["Yes", "No"])
work_interfere = st.selectbox("Does a mental condition interfere with your work?", ["Never", "Rarely", "Sometimes", "Often", "Don't know"])
no_employees = st.selectbox("Number of employees", ["0","1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
tech_company = st.selectbox("Is your employer a tech company?", ["Select", "College Student","Yes", "No"])
benefits = st.selectbox("Does your employer provide mental health benefits?", ["Yes", "No", "Don't know"])
care_options = st.selectbox("Do you know the care options your employer provides?", ["Yes", "No", "Not sure"])
wellness_program = st.selectbox("Has your employer discussed mental health as part of a wellness program?", ["Yes", "No", "Don't know"])
seek_help = st.selectbox("Resources for seeking help available?", ["Yes", "No", "Don't know"])
anonymity = st.selectbox("Is your anonymity protected?", ["Yes", "No", "Don't know"])
leave = st.selectbox("Ease of taking medical leave for mental health?", ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"])
mental_health_consequence = st.selectbox("Negative consequence for discussing mental health?", ["Yes", "No", "Maybe"])
phys_health_consequence = st.selectbox("Negative consequence for discussing physical health?", ["Yes", "No", "Maybe"])
coworkers = st.selectbox("Willing to discuss with coworkers?", ["Yes", "No", "Some of them"])
supervisor = st.selectbox("Willing to discuss with supervisor?", ["Yes", "No", "Some of them"])
mental_health_interview = st.selectbox("Would bring up mental health in interview?", ["Yes", "No", "Maybe"])
phys_health_interview = st.selectbox("Would bring up physical health in interview?", ["Yes", "No", "Maybe"])
mental_vs_physical = st.selectbox("Employer takes mental health as seriously as physical?", ["Yes", "No", "Don't know"])
obs_consequence = st.selectbox("Observed consequences for others?", ["Yes", "No"])
comments = st.text_area("Any additional comments?", "")

# --- Manual One-Hot Encoding into Template ---
input_df = template.copy()
input_df["Age"] = age
input_df["comments"] = 0  # Ignored by model

def set_onehot(col_prefix, value):
    col_name = f"{col_prefix}_{value}"
    if col_name in input_df.columns:
        input_df.loc[:, input_df.columns.str.startswith(col_prefix + "_")] = 0
        input_df[col_name] = 1

set_onehot("Gender", gender)
set_onehot("Country", country)
set_onehot("self_employed", self_employed)
set_onehot("family_history", family_history)
set_onehot("treatment", treatment)
set_onehot("work_interfere", work_interfere)
set_onehot("no_employees", no_employees)
set_onehot("remote_work", remote_work)
set_onehot("tech_company", tech_company)
set_onehot("benefits", benefits)
set_onehot("care_options", care_options)
set_onehot("wellness_program", wellness_program)
set_onehot("seek_help", seek_help)
set_onehot("anonymity", anonymity)
set_onehot("leave", leave)
set_onehot("mental_health_consequence", mental_health_consequence)
set_onehot("phys_health_consequence", phys_health_consequence)
set_onehot("coworkers", coworkers)
set_onehot("supervisor", supervisor)
set_onehot("mental_health_interview", mental_health_interview)
set_onehot("phys_health_interview", phys_health_interview)
set_onehot("mental_vs_physical", mental_vs_physical)
set_onehot("obs_consequence", obs_consequence)

# --- Reorder Columns to Match Model ---
input_df = input_df.reindex(columns=model_columns, fill_value=0)
input_df = input_df.astype(np.float32)

# --- Predict ---
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction > 0.5:
        st.error("⚠️ Likely to need mental health support.")
    else:
        st.success("✅ Low risk of needing mental health treatment.")
