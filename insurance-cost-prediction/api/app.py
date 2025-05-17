from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('models/trained_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df['BMI'] = df['Weight'] / ((df['Height']/100)**2)
    columns = ['Age','Diabetes','BloodPressureProblems','AnyTransplants','AnyChronicDiseases',
               'Height','Weight','KnownAllergies','HistoryOfCancerInFamily',
               'NumberOfMajorSurgeries','BMI']
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    df = df[columns]
    pred = model.predict(df)[0]
    return jsonify({'predicted_premium': round(pred, 2)})

if __name__ == "__main__":
    app.run(debug=True)
