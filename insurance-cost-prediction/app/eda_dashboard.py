import sys
import os

# Add project root to sys.path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway, chi2_contingency, zscore
import joblib
from src.modeling import train_and_save_models, get_project_root

st.set_page_config(page_title="Insurance EDA & Prediction Dashboard", layout="wide")

# --- Custom CSS for professional look and static header/footer with pastel purple theme ---
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', Arial, sans-serif;
        background-color: #f8f6ff;
        font-size: 16px;
    }
    .main {background-color: #f8f6ff;}
    .stApp {padding-top: 70px; padding-bottom: 50px;}
    header {
        position: fixed; top: 0; left: 0; right: 0; height: 60px;
        background: #a084ee; color: #22223b; z-index: 9999;
        display: flex; align-items: center; padding-left: 32px;
        font-size: 2.1rem; font-weight: 700; letter-spacing: 1px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    footer {
        position: fixed; bottom: 0; left: 0; right: 0; height: 40px;
        background: #ffe066; color: #22223b; text-align: center;
        line-height: 40px; font-size: 1.05rem; z-index: 9999;
        box-shadow: 0 -2px 8px rgba(0,0,0,0.04);
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #e7c6ff;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.08rem;
        padding: 7px 20px 7px 20px;
        color: #5f6f94;
    }
    .stTabs [aria-selected="true"] {
        background: #a084ee !important;
        color: #fff !important;
        border-radius: 8px 8px 0 0;
    }
    .stButton>button {
        background: #a084ee;
        color: #fff;
        font-size: 1.05rem;
        padding: 0.45rem 1.2rem;
        border-radius: 6px;
    }
    .prediction-box {
        background-color: #e7c6ff;
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        font-size: 1.15rem;
        font-weight: 600;
        color: #22223b;
        text-align: center;
        border: 1.2px solid #a084ee;
    }
    .stDataFrame {font-size: 0.95rem;}
    .stExpander {font-size: 1rem;}
    section.main > div:first-child {padding-top: 10px;}
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #fffbe6 !important;
    }
    </style>
    <header>📊 Insurance Cost Prediction: EDA & Insights</header>
    <footer>Made with ❤️ by Tanusree | Insurance Cost Predictor</footer>
""", unsafe_allow_html=True)

# --- Logo (show exactly before the title) ---
logo_path = os.path.join(get_project_root(), "resources", "logo.png")
if os.path.exists(logo_path):
    st.image(logo_path, width=110)

st.markdown("<h1 style='font-size:1.7rem; margin-bottom:0.2em;'>Insurance Cost Prediction Dashboard</h1>", unsafe_allow_html=True)

# --- Load Data ---
data_path = os.path.join(get_project_root(), "data", "insurance.csv")
df = pd.read_csv(data_path)
df['BMI'] = df['Weight'] / ((df['Height']/100)**2)

# --- Health Condition Columns: Auto-detect for robustness ---
def find_col(possibilities):
    for col in df.columns:
        for p in possibilities:
            if p.lower() in col.lower():
                return col
    return None

health_cols = [
    find_col(['diabetes']),
    find_col(['blood pressure']),
    find_col(['transplant']),
    find_col(['chronic']),
    find_col(['allergies']),
    find_col(['cancer'])
]
health_labels = ['Diabetes', 'BP Problems', 'Transplant', 'Chronic', 'Allergies', 'Cancer History']

# --- Sidebar Filters ---
st.sidebar.header("🔎 Filter Data")
age_range = st.sidebar.slider("Age Range", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
weight_range = st.sidebar.slider("Weight Range", int(df['Weight'].min()), int(df['Weight'].max()), (int(df['Weight'].min()), int(df['Weight'].max())))
height_range = st.sidebar.slider("Height Range", int(df['Height'].min()), int(df['Height'].max()), (int(df['Height'].min()), int(df['Height'].max())))
diabetes = st.sidebar.selectbox("Diabetes", options=["All", 0, 1])
bp = st.sidebar.selectbox("Blood Pressure Problems", options=["All", 0, 1])
chronic = st.sidebar.selectbox("Chronic Diseases", options=["All", 0, 1])
allergies = st.sidebar.selectbox("Known Allergies", options=["All", 0, 1])
cancer = st.sidebar.selectbox("History of Cancer in Family", options=["All", 0, 1])

# --- Apply Filters ---
filtered_df = df[
    (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
    (df['Weight'] >= weight_range[0]) & (df['Weight'] <= weight_range[1]) &
    (df['Height'] >= height_range[0]) & (df['Height'] <= height_range[1])
]
col_map = {
    'Diabetes': health_cols[0],
    'Blood Pressure Problems': health_cols[1],
    'Chronic Diseases': health_cols[3],
    'Known Allergies': health_cols[4],
    'History of Cancer in Family': health_cols[5]
}
for label, val in zip(['Diabetes', 'Blood Pressure Problems', 'Chronic Diseases', 'Known Allergies', 'History of Cancer in Family'],
                      [diabetes, bp, chronic, allergies, cancer]):
    col = col_map[label]
    if val != "All" and col:
        filtered_df = filtered_df[filtered_df[col] == int(val)]

# --- Tabs for Dashboard Sections ---
tab_pred, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💡 Premium Estimator", "📈 Summary", "💰 Premium Analysis", "⚕️ Risk Factors", "👥 Demographics", "📑 EDA & Hypothesis"
])

# --- Premium Estimator (Model Integration) ---
with tab_pred:
    st.markdown("<h2 style='font-size:1.2rem;'>💡 Predict Your Insurance Premium</h2>", unsafe_allow_html=True)
    results = train_and_save_models(df)
    best_model_name = min(results, key=lambda k: results[k]["rmse"])
    model_path = os.path.join(get_project_root(), "models", f"{best_model_name}_model.pkl")
    model = joblib.load(model_path)

    with st.expander("📝 Enter Your Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 66, 30)
            diabetes_val = st.selectbox("Diabetes", [0, 1])
            bp_val = st.selectbox("Blood Pressure Problems", [0, 1])
            transplant = st.selectbox("Any Transplants", [0, 1])
            chronic_val = st.selectbox("Any Chronic Diseases", [0, 1])
        with col2:
            height = st.slider("Height (cm)", 145, 188, 170)
            weight = st.slider("Weight (kg)", 51, 132, 70)
            allergies_val = st.selectbox("Known Allergies", [0, 1])
            cancer_val = st.selectbox("History of Cancer in Family", [0, 1])
            surgeries = st.slider("Number of Major Surgeries", 0, 3, 0)

        bmi = weight / ((height/100) ** 2)

        if st.button("Predict Premium", key="predict_button"):
            input_data = np.array([[age, diabetes_val, bp_val, transplant, chronic_val, height, weight,
                                    allergies_val, cancer_val, surgeries, bmi]])
            premium = model.predict(input_data)[0]
            st.markdown(
                f"<div class='prediction-box'><h3>Estimated Premium Price: ₹{premium:,.2f}</h3></div>",
                unsafe_allow_html=True
            )

# --- 1. Summary Statistics Dashboard ---
with tab1:
    st.markdown("<h2 style='font-size:1.2rem;'>📈 Summary Statistics Dashboard</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Premium", f"₹{filtered_df['PremiumPrice'].mean():,.2f}")
    col2.metric("Average Age", f"{filtered_df['Age'].mean():.1f} years")
    col3.metric("Average BMI", f"{filtered_df['BMI'].mean():.1f}")

    st.markdown("<h4 style='font-size:1.02rem;'>Count of Individuals by Health Conditions</h4>", unsafe_allow_html=True)
    health_counts = [filtered_df[c].sum() if c in filtered_df.columns else 0 for c in health_cols]
    fig, ax = plt.subplots(figsize=(4.5,2.1))
    sns.barplot(x=health_labels, y=health_counts, palette="Purples", ax=ax)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Individuals with Health Conditions", fontsize=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    st.pyplot(fig)

    st.markdown("<h4 style='font-size:1.02rem;'>Distribution of Key Health Conditions</h4>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(4.5,2.1))
    health_df = pd.DataFrame({'Condition': health_labels, 'Count': health_counts})
    ax2.pie(health_df['Count'], labels=health_df['Condition'], autopct='%1.1f%%', colors=sns.color_palette("Purples", len(health_labels)))
    ax2.set_title("Proportion of Health Conditions", fontsize=10)
    st.pyplot(fig2)

# --- 2. Premium Pricing Analysis Dashboard ---
with tab2:
    st.markdown("<h2 style='font-size:1.2rem;'>💰 Premium Pricing Analysis Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-size:1.02rem;'>Premium Distribution</h4>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4.5,2.1))
    sns.histplot(filtered_df['PremiumPrice'], bins=30, kde=True, color="#a084ee", ax=ax)
    ax.set_xlabel("Premium Price", fontsize=9)
    ax.set_title("Distribution of Premium Prices", fontsize=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    st.pyplot(fig)

    st.markdown("<h4 style='font-size:1.02rem;'>Premiums by Age Group and Health Factors</h4>", unsafe_allow_html=True)
    age_bins = pd.cut(filtered_df['Age'], bins=[17,25,35,45,55,65,100], labels=['18-25','26-35','36-45','46-55','56-65','66+'])
    diabetes_col = health_cols[0]
    grouped = filtered_df.groupby([age_bins, diabetes_col])['PremiumPrice'].mean().reset_index()
    grouped.columns = ['Age Group', 'Diabetes', 'PremiumPrice']
    fig2, ax2 = plt.subplots(figsize=(4.5,2.1))
    sns.barplot(x='Age Group', y='PremiumPrice', hue='Diabetes', data=grouped, palette="Purples", ax=ax2)
    ax2.set_ylabel("Avg Premium", fontsize=9)
    ax2.set_title("Avg Premium by Age Group & Diabetes", fontsize=10)
    ax2.tick_params(axis='x', labelsize=8)
    ax2.legend(title="Diabetes", fontsize=8, title_fontsize=9)
    st.pyplot(fig2)

    st.markdown("<h4 style='font-size:1.02rem;'>Correlation Heatmap</h4>", unsafe_allow_html=True)
    corr = filtered_df[['Age','Height','Weight','BMI','NumberOfMajorSurgeries','PremiumPrice']].corr()
    fig3, ax3 = plt.subplots(figsize=(3.5,2.1))
    sns.heatmap(corr, annot=True, cmap="Purples", ax=ax3, annot_kws={"size":8})
    ax3.set_title("Correlation Matrix", fontsize=10)
    st.pyplot(fig3)

# --- 3. Risk Factors Analysis Dashboard ---
with tab3:
    st.markdown("<h2 style='font-size:1.2rem;'>⚕️ Risk Factors Analysis Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-size:1.02rem;'>Surgical Impact on Premiums</h4>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4.5,2.1))
    sns.boxplot(x='NumberOfMajorSurgeries', y='PremiumPrice', data=filtered_df, palette="Purples", ax=ax)
    ax.set_title("Premiums by Number of Major Surgeries", fontsize=10)
    ax.set_xlabel("Number of Major Surgeries", fontsize=9)
    ax.set_ylabel("Premium Price", fontsize=9)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    st.pyplot(fig)

    st.markdown("<h4 style='font-size:1.02rem;'>Impact of Chronic Conditions</h4>", unsafe_allow_html=True)
    chronic_col = health_cols[3]
    transplant_col = health_cols[2]
    fig2, ax2 = plt.subplots(figsize=(4.5,2.1))
    chronic_group = filtered_df.groupby([chronic_col, transplant_col])['PremiumPrice'].mean().reset_index()
    chronic_group.columns = ['ChronicDiseases', 'Transplant', 'PremiumPrice']
    chronic_group['ChronicDiseases'] = chronic_group['ChronicDiseases'].map({0:'No Chronic', 1:'Chronic'})
    chronic_group['Transplant'] = chronic_group['Transplant'].map({0:'No Transplant', 1:'Transplant'})
    sns.barplot(x='ChronicDiseases', y='PremiumPrice', hue='Transplant', data=chronic_group, palette="Purples", ax=ax2)
    ax2.set_title("Premium by Chronic Disease & Transplant", fontsize=10)
    ax2.legend(title="Transplant", fontsize=8, title_fontsize=9)
    ax2.tick_params(axis='x', labelsize=8)
    st.pyplot(fig2)

    st.markdown("<h4 style='font-size:1.02rem;'>Allergies and Family History Influence</h4>", unsafe_allow_html=True)
    allergy_col = health_cols[4]
    cancer_col = health_cols[5]
    fig3, ax3 = plt.subplots(figsize=(4.5,2.1))
    allergy_group = filtered_df.groupby([allergy_col, cancer_col])['PremiumPrice'].mean().reset_index()
    allergy_group.columns = ['KnownAllergies', 'HistoryOfCancerInFamily', 'PremiumPrice']
    allergy_group['KnownAllergies'] = allergy_group['KnownAllergies'].map({0:'No Allergy', 1:'Allergy'})
    allergy_group['HistoryOfCancerInFamily'] = allergy_group['HistoryOfCancerInFamily'].map({0:'No Cancer', 1:'Cancer'})
    sns.barplot(x='KnownAllergies', y='PremiumPrice', hue='HistoryOfCancerInFamily', data=allergy_group, palette="Purples", ax=ax3)
    ax3.set_title("Premium by Allergy & Cancer History", fontsize=10)
    ax3.legend(title="Cancer History", fontsize=8, title_fontsize=9)
    ax3.tick_params(axis='x', labelsize=8)
    st.pyplot(fig3)

# --- 4. Demographic Insights Dashboard ---
with tab4:
    st.markdown("<h2 style='font-size:1.2rem;'>👥 Demographic Insights Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-size:1.02rem;'>Premiums by Height and Weight (BMI)</h4>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4.5,2.1))
    scatter = ax.scatter(filtered_df['BMI'], filtered_df['PremiumPrice'], c=filtered_df['Age'], cmap='Purples', alpha=0.7, s=18)
    ax.set_xlabel("BMI", fontsize=9)
    ax.set_ylabel("Premium Price", fontsize=9)
    ax.set_title("Premiums by BMI (Colored by Age)", fontsize=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Age', fontsize=8)
    st.pyplot(fig)

    # Geographical Analysis (if region/area column exists)
    region_col = None
    for col in df.columns:
        if 'region' in col.lower() or 'area' in col.lower():
            region_col = col
            break
    if region_col:
        st.markdown("<h4 style='font-size:1.02rem;'>Geographical Breakdown of Premiums</h4>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(4.5,2.1))
        region_group = filtered_df.groupby(region_col)['PremiumPrice'].mean().reset_index()
        sns.barplot(x=region_col, y='PremiumPrice', data=region_group, palette="Purples", ax=ax2)
        ax2.set_title("Avg Premium by Region", fontsize=10)
        ax2.tick_params(axis='x', labelsize=8)
        ax2.tick_params(axis='y', labelsize=8)
        st.pyplot(fig2)
    else:
        st.info("No region/area column found for geographical analysis.")

# --- 5. EDA & Hypothesis Testing ---
with tab5:
    st.markdown("<h2 style='font-size:1.2rem;'>📑 EDA & Hypothesis Testing</h2>", unsafe_allow_html=True)

    st.markdown("<h4 style='font-size:1.02rem;'>Outlier Detection (Z-score > 3)</h4>", unsafe_allow_html=True)
    num_cols = ['Age','Height','Weight','BMI','NumberOfMajorSurgeries','PremiumPrice']
    z_scores = np.abs(zscore(filtered_df[num_cols]))
    outliers = (z_scores > 3).any(axis=1)
    st.write(f"Number of outliers detected: {outliers.sum()}")
    st.dataframe(filtered_df[outliers][num_cols].head(10))

    st.markdown("<h4 style='font-size:1.02rem;'>T-test: Premiums for Diabetes vs. Non-Diabetes</h4>", unsafe_allow_html=True)
    diabetes_col = health_cols[0]
    tstat, pval = ttest_ind(
        filtered_df[filtered_df[diabetes_col]==1]['PremiumPrice'],
        filtered_df[filtered_df[diabetes_col]==0]['PremiumPrice'],
        equal_var=False
    )
    st.write(f"T-statistic: {tstat:.2f}, p-value: {pval:.4f}")
    if pval < 0.05:
        st.success("Premiums are significantly different for diabetes vs. non-diabetes.")
    else:
        st.info("No significant difference in premiums for diabetes vs. non-diabetes.")

    st.markdown("<h4 style='font-size:1.02rem;'>ANOVA: Premiums by Number of Surgeries</h4>", unsafe_allow_html=True)
    groups = [group['PremiumPrice'].values for name, group in filtered_df.groupby('NumberOfMajorSurgeries')]
    fstat, pval = f_oneway(*groups)
    st.write(f"F-statistic: {fstat:.2f}, p-value: {pval:.4f}")
    if pval < 0.05:
        st.success("Premiums differ significantly by number of surgeries.")
    else:
        st.info("No significant difference in premiums by number of surgeries.")

    st.markdown("<h4 style='font-size:1.02rem;'>Chi-square: Chronic Disease vs. Cancer History</h4>", unsafe_allow_html=True)
    chronic_col = health_cols[3]
    cancer_col = health_cols[5]
    contingency = pd.crosstab(filtered_df[chronic_col], filtered_df[cancer_col])
    chi2, pval, _, _ = chi2_contingency(contingency)
    st.write(f"Chi-square: {chi2:.2f}, p-value: {pval:.4f}")
    if pval < 0.05:
        st.success("Chronic disease and cancer history are associated.")
    else:
        st.info("No significant association between chronic disease and cancer history.")

    st.markdown("<h4 style='font-size:1.02rem;'>Regression Analysis: Predicting Premiums</h4>", unsafe_allow_html=True)
    import statsmodels.api as sm
    X = filtered_df[['Age','BMI','NumberOfMajorSurgeries',health_cols[0],health_cols[3],health_cols[4]]]
    X = sm.add_constant(X)
    y = filtered_df['PremiumPrice']
    model = sm.OLS(y, X).fit()
    st.write(model.summary())

    st.markdown("""
    **Insights & Recommendations:**
    - Diabetes, chronic diseases, and number of surgeries are strong predictors of higher premiums.
    - Consider targeted wellness programs for high-risk groups.
    - Policy adjustments could include discounts for healthy BMI or no chronic conditions.
    """)