def add_bmi(df):
    df = df.copy()
    df['BMI'] = df['Weight'] / ((df['Height']/100)**2)
    return df
