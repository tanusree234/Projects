import sys
import os

import streamlit as st

st.set_page_config(page_title="Insurance Cost Prediction", layout="wide")

# Add project root to sys.path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from src.modeling import train_and_save_models, get_project_root
from src.visualization import plot_actual_vs_pred, plot_model_comparison

# --- Custom CSS for professional look and static header/footer ---
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', Arial, sans-serif;
        background-color: #f8fafc;
    }
    .main {
        background-color: #f8fafc;
    }
    .stApp {
        padding-top: 70px;
        padding-bottom: 50px;
    }
    header {
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 60px;
        background: #a5d8ff;
        color: #22223b;
        z-index: 9999;
        display: flex;
        align-items: center;
        padding-left: 32px;
        font-size: 2rem;
        font-weight: 600;
        letter-spacing: 1px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    footer {
        position: fixed;
        bottom: 0; left: 0; right: 0;
        height: 40px;
        background: #b2f2bb;
        color: #22223b;
        text-align: center;
        line-height: 40px;
        font-size: 1rem;
        z-index: 9999;
        box-shadow: 0 -2px 8px rgba(0,0,0,0.04);
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #e7c6ff;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        padding: 8px 24px 8px 24px;
        color: #5f6f94;
    }
    .stTabs [aria-selected="true"] {
        background: #b2f2bb !important;
        color: #22223b !important;
        border-radius: 8px 8px 0 0;
    }
    .stButton>button {
        background: #a5d8ff;
        color: #22223b;
        font-size: 1.1rem;
        padding: 0.5rem 1.5rem;
        border-radius: 6px;
    }
    .prediction-box {
        background-color: #e7c6ff;
        padding: 18px;
        border-radius: 10px;
        margin-top: 18px;
        font-size: 1.3rem;
        font-weight: 600;
        color: #22223b;
        text-align: center;
        border: 1.5px solid #b2f2bb;
    }
    .stDataFrame {
        font-size: 1rem;
    }
    .stExpander {
        font-size: 1rem;
    }
    </style>
    <header>🚗 Insurance Cost Prediction Portal</header>
    <footer>Made with ❤️ by Your Team | Scalar Study Materials</footer>
""", unsafe_allow_html=True)

# --- Logo (show exactly before the title) ---
logo_path = os.path.join(get_project_root(), "resources", "logo.png")
if os.path.exists(logo_path):
    st.image(logo_path, width=120)

st.title("Insurance Cost Prediction App")

# Data loading
data_path = os.path.join(get_project_root(), "data", "insurance.csv")
df = pd.read_csv(data_path)
df['BMI'] = df['Weight'] / ((df['Height']/100)**2)

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["💡 Estimated Cost", "📁 Data", "📊 Model Comparison"])

with tab2:
    st.header("📁 Data Preview")
    st.dataframe(df.head(20), height=350)
    st.markdown("**Columns:** " + ", ".join(df.columns))

with tab3:
    st.header("📊 Model Performance Comparison")
    results = train_and_save_models(df)
    try:
        plot_model_comparison(results)
    except Exception:
        st.info("Add y_test and y_pred to results in modeling.py to enable comparison plots.")

    with st.expander("🔬 Detailed Model Results", expanded=True):
        for model_name, res in results.items():
            st.subheader(f"{model_name}")
            st.write(f"**RMSE:** {res['rmse']:.2f} | **R²:** {res['r2']:.3f}")
            if "y_test" in res and "y_pred" in res:
                plot_actual_vs_pred(res["y_test"], res["y_pred"], model_name)

with tab1:
    st.header("💡 Predict Your Insurance Premium")
    # Load the best model (lowest RMSE)
    best_model_name = min(results, key=lambda k: results[k]["rmse"])
    model_path = os.path.join(get_project_root(), "models", f"{best_model_name}_model.pkl")
    model = joblib.load(model_path)

    with st.expander("📝 Enter Your Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 66, 30)
            diabetes = st.selectbox("Diabetes", [0, 1])
            bp = st.selectbox("Blood Pressure Problems", [0, 1])
            transplant = st.selectbox("Any Transplants", [0, 1])
            chronic = st.selectbox("Any Chronic Diseases", [0, 1])
        with col2:
            height = st.slider("Height (cm)", 145, 188, 170)
            weight = st.slider("Weight (kg)", 51, 132, 70)
            allergies = st.selectbox("Known Allergies", [0, 1])
            cancer = st.selectbox("History of Cancer in Family", [0, 1])
            surgeries = st.slider("Number of Major Surgeries", 0, 3, 0)

        bmi = weight / ((height/100) ** 2)

        if st.button("Predict Premium", key="predict_button"):
            input_data = np.array([[age, diabetes, bp, transplant, chronic, height, weight,
                                    allergies, cancer, surgeries, bmi]])
            premium = model.predict(input_data)[0]
            st.markdown(
                f"<div class='prediction-box'><h3>Estimated Premium Price: ₹{premium:,.2f}</h3></div>",
                unsafe_allow_html=True
            )