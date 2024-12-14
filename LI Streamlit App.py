import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

@st.cache_data
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def clean_sm(x):
    return np.where(x == 1, 1, 0)

st.title("LinkedIn User Prediction App")

st.markdown("""
This app predicts whether a person uses LinkedIn and provides the probability of LinkedIn usage 
based on user inputs.
""")

st.header("User Input Features")
income = st.number_input("Income (1-9 scale)", min_value=1, max_value=9, value=5)
education = st.number_input("Education Level (1-8 scale)", min_value=1, max_value=8, value=4)
parent = st.radio("Is the person a parent?", ["Yes", "No"])
married = st.radio("Married?", ["Yes", "No"])
female = st.radio("Is the person female?", ["Yes", "No"])
age = st.number_input("Age", min_value=18, max_value=98, value=30)

parent_bin = 1 if parent == "Yes" else 0
married_bin = 1 if married == "Yes" else 0
female_bin = 1 if female == "Yes" else 0

input_data = pd.DataFrame({
    "income": [income],
    "education": [education],
    "parent": [parent_bin],
    "married": [married_bin],
    "female": [female_bin],
    "age": [age]
})

model = load_model()

if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.success(f"The person is likely a LinkedIn user with a probability of {probability:.2%}.")
    else:
        st.error(f"The person is unlikely to use LinkedIn. Probability: {probability:.2%}.")

import pickle
with open("model.pkl", "wb") as file:
    pickle.dump(log_reg, file)
