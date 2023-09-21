import streamlit as st
import pandas as pd
import numpy as np
from collections import ChainMap
import joblib
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler,LabelEncoder



pickled_model = pickle.load(open('model.pkl', 'rb'))
st.title("Developer's Salary Prediction in 2022 📈💲")
st.sidebar.markdown('''# :orange[Features]   :ballot_box_with_check:''')
st.sidebar.write("""### We need some information to predict the salary 👇""")
countries = (
    "United States of America",
    "Iran, Islamic Republic of...",
    "India",
    "United Kingdom of Great Britain and Northern Ireland",
    "Germany",
    "Canada",
    "Brazil",
    "France",
    "Spain",
    "Australia",
    "Netherlands",
    "Poland",
    "Italy",
    "Russian Federation",
    "Sweden",
    "Switzerland",
    "Israel",
    "Austria",
    "Portugal",
    "Denmark",
    "Turkey",
    "Belgium",
    "Norway",
    "Finland",
    "Greece",
    "Czech Republic",
    "New Zealand",
    "Mexico",
    "South Africa",
    "Pakistan"
)
education = (
    "Less than a Bachelors",
    "Bachelor’s degree",
    "Master’s degree"
)
st.sidebar.divider()
country = st.sidebar.selectbox("Country", countries)
st.sidebar.divider()
education = st.sidebar.selectbox("Education Level 👨‍🎓", education)
st.sidebar.divider()
expericence = st.sidebar.slider("Years of Experience 👴", 0, 50, 3)
st.sidebar.divider()
gender = st.sidebar.radio('Gender :man-pouting::woman-pouting:', ["Man", "Woman", "Other"])
st.sidebar.divider()
age = st.sidebar.selectbox("Age", ["25-34 years old", "35-44 years old ",
                                   "18-24 years old", "45-54 years old", "55-64 years old", "Other"])
columns = ['Country', 'EdLevel', 'YearsCodePro', 'Age', 'Gender']

ok = st.sidebar.button("Calculate Salary")
st.divider()
if ok:
    X_new = np.array([country, education, expericence, age, gender])
    X_new_df = pd.DataFrame([X_new], columns=columns)
    salary = pickled_model.predict(X_new_df)
    st.subheader(f"The estimated salary is:  ${salary[0]:.2f}")