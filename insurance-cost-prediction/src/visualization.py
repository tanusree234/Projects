import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_actual_vs_pred(y_test, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(y_test, y_pred, alpha=0.6, c=y_pred, cmap='coolwarm')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual Premium Price")
    ax.set_ylabel("Predicted Premium Price")
    ax.set_title(f"Actual vs Predicted: {model_name}", fontsize=10)
    st.pyplot(fig)

def plot_model_comparison(results):
    fig, ax = plt.subplots(figsize=(2, 1))
    model_names = list(results.keys())
    rmses = [results[m]["rmse"] for m in model_names]
    sns.barplot(x=model_names, y=rmses, palette="coolwarm", ax=ax, orient='v', hue=0.5)
    ax.set_ylabel("RMSE (Lower is better)", fontsize=6)
    ax.set_title("Model RMSE Comparison", fontsize=8)
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    st.pyplot(fig)