import streamlit as st
import joblib
import numpy as np

model = joblib.load('models/trained_model.pkl')

st.title("Insurance Premium Estimator")

age = st.slider("Age", 18, 66, 30)
diabetes = st.selectbox("Diabetes", [0, 1])
bp = st.selectbox("Blood Pressure Problems", [0, 1])
transplant = st.selectbox("Any Transplants", [0, 1])
chronic = st.selectbox("Any Chronic Diseases", [0, 1])
height = st.slider("Height (cm)", 145, 188, 170)
weight = st.slider("Weight (kg)", 51, 132, 70)
allergies = st.selectbox("Known Allergies", [0, 1])
cancer = st.selectbox("History of Cancer in Family", [0, 1])
surgeries = st.slider("Number of Major Surgeries", 0, 3, 0)

bmi = weight / ((height/100) ** 2)

if st.button("Predict Premium"):
    input_data = np.array([[age, diabetes, bp, transplant, chronic, height, weight,
                            allergies, cancer, surgeries, bmi]])
    premium = model.predict(input_data)[0]
    st.success(f"Estimated Premium Price: ₹{premium:,.2f}")
