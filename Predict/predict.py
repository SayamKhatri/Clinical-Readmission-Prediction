import pandas as pd
import joblib
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Training.Feature_engineering import build_features


rf          = joblib.load(os.path.join(PROJECT_ROOT,'Training', 'rf_model.pkl'))
rf_features = joblib.load(os.path.join(PROJECT_ROOT, 'Training', 'rf_features.pkl'))

RF_THRESH = 0.512

def predict(df_new):
    """
    Input: df_new (raw DataFrame with same columns as the training CSV)
    Output: DataFrame with risk scores and binary flags
    """
    X_all, _, _ = build_features(df_new)

    X_sub = X_all[rf_features]

    probs = rf.predict_proba(X_sub)[:,1]
    flags = (probs >= RF_THRESH).astype(int)

    return pd.DataFrame({
        'risk_score': probs,
        'flag_readmit': flags
    }, index=df_new.index)

if __name__=='__main__':
    df = pd.read_csv('diabetic_data.csv')
    result = predict(df)
    print(result.head())

