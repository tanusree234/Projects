import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import joblib
import os

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_and_save_models(df):
    X = df.drop(['PremiumPrice'], axis=1)
    y = df['PremiumPrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "linear_regression": LinearRegression()
    }

    os.makedirs(os.path.join(get_project_root(), "models"), exist_ok=True)
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        joblib.dump(model, os.path.join(get_project_root(), "models", f"{name.replace(' ', '_').lower()}_model.pkl"))
        results[name] = {"rmse": rmse, "r2": r2}
        print(f"{name} saved to models/{name}_model.pkl | RMSE: {rmse:.2f} | R2: {r2:.3f}")

    return results