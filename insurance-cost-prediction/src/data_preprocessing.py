import pandas as pd
from sklearn.impute import SimpleImputer

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df = df.copy()
    num_cols = ['Age', 'Height', 'Weight', 'NumberOfMajorSurgeries']
    imp = SimpleImputer(strategy='mean')
    df[num_cols] = imp.fit_transform(df[num_cols])
    bin_cols = ['Diabetes', 'BloodPressureProblems', 'AnyTransplants',
                'AnyChronicDiseases', 'KnownAllergies', 'HistoryOfCancerInFamily']
    df[bin_cols] = df[bin_cols].astype(int)
    return df
