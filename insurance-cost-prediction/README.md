# Insurance Cost Prediction

## Overview
Predict insurance premium costs using health and demographic information, powered by machine learning.

## Structure
- EDA, feature engineering, ML modeling in `src/`
- Data in `data/`
- API in `api/`
- Web App in `app/`
- Models in `models/`

## Run
1. Install requirements: `pip install -r requirements.txt`
2. Train model: see `src/modeling.py`.
3. Run Flask API: `cd api && python app.py`
4. Run Streamlit app: `cd app && streamlit run streamlit_app.py`
