from Feature_engineering import build_features
import numpy as np
import pandas as pd
from collections import Counter
import joblib

import pandas as pd
df = pd.read_csv('/Users/samkhatri/Desktop/Data Science Projects/Clinical-Readmission-Prediction/Data/diabetic_data.csv')


def main():
    X_all, y, _ = build_features(df)

    nn_features = [
        'time_in_hospital','num_lab_procedures','num_procedures',
        'num_medications','number_outpatient','number_emergency',
        'number_inpatient','number_diagnoses','insulin_prescribed',
        'insulin_up','insulin_down','num_oral_meds_prescribed',
        'max_glu_serum_ord','A1Cresult_ord','admType_Elective',
        'admType_Emergency','admType_Urgent','admSrc_ER',
        'admSrc_PhysicianReferral','disp_Home','disp_HomeHealth',
        'disp_HospiceFacility','disp_OtherCare','disp_OtherHospital',
        'disp_RehabFacility','disp_SNF','age_50to74','age_75plus',
        'race_AfricanAmerican','race_Caucasian',
        'payer_MC','payer_Unknown','medspec_Cardiology',
        'medspec_Unknown','age_insulin_risk','complex_stay',
        'charlson_index','prior_admissions'
    ]



    X = X_all[nn_features].values
    y = y.values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, classification_report

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)


    y_rf_prob = rf.predict_proba(X_test)[:,1]
    rf_auc = roc_auc_score(y_test, y_rf_prob)
    print(f"RF Test AUC: {rf_auc:.3f}")

    rf_thresh = 0.512
    y_rf_pred = (y_rf_prob >= rf_thresh).astype(int)
    print("RF classification report (th=0.512):")
    print(classification_report(y_test, y_rf_pred, digits=3))


    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf


    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)


    nn = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_s.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    nn.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(1e-3),
        loss='binary_crossentropy'
    )


    counts = Counter(y_train)
    total = len(y_train)
    class_weight = {
        0: total / (2 * counts[0]),
        1: total / (2 * counts[1])
    }


    nn.fit(
        X_train_s, y_train,
        epochs=20,
        batch_size=256,
        class_weight=class_weight,
        validation_split=0.1,
        verbose=2
    )


    y_nn_prob = nn.predict(X_test_s).ravel()
    nn_auc = roc_auc_score(y_test, y_nn_prob)
    print(f"NN Test AUC: {nn_auc:.3f}")

    nn_thresh = 0.512
    y_nn_pred = (y_nn_prob >= nn_thresh).astype(int)
    print("NN classification report (th=0.512):")
    print(classification_report(y_test, y_nn_pred, digits=3))


    joblib.dump(rf, 'rf_model.pkl')
    joblib.dump(nn_features, 'rf_features.pkl')


if __name__=="__main__":
    main()
