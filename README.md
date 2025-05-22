# Clinical-Readmission-Prediction

An ML tool to identify high‐risk patients and help hospitals proactively reduce 30-day readmissions.


# Why This Matters

Hospital readmission rates are closely monitored by healthcare agencies and can trigger financial penalties when they exceed acceptable thresholds. For patients, unplanned 30-day readmissions not only harm patient outcomes but also drive up costs and impact hospital quality ratings. This project demonstrates how a data-driven approach can flag patients at greatest risk—enabling care teams to intervene early, improve post-discharge support, and ultimately lower readmission penalties.


# Approach & Pipeline

1. **Feature Engineering**

   * **Medication changes**: Count of oral diabetes meds started, stopped, increased or decreased.
   * **Insulin patterns**: Binary flags for prescription, dosage up/down/steady.
   * **Lab ordinals**: Map `max_glu_serum` and `A1Cresult` into 0–3 severity scores.
   * **Admission/Discharge grouping**: Collapse rare categories into “Other” and one-hot encode the top event types.
   * **Diagnosis lift**: Identify top 10 primary diagnoses most strongly associated with readmission “lift” and encode them.
   * **Interaction features**:

     * `age_insulin_risk` (elderly on insulin)
     * `unstable_patient` (prior inpatient admissions + med changes)
     * `complex_stay` (long hospital stay + many procedures)
   * **Comorbidity**: Charlson index from ICD-9 diagnosis codes.
   * **Prior utilization**: Count of previous admissions per patient.

2. **Modeling**

   * **Random Forest** (primary): balanced class weights, tuned threshold (0.512) to optimize F1/recall trade-off.
   * **Neural Network** (comparison): two hidden layers, class-weighted binary cross-entropy.

3. **Evaluation**

   * **ROC-AUC**: RF ≈ 0.78, NN ≈ 0.78 on held-out test set.
   * **Recall (readmit within 30 days)**: \~ 75% at optimized threshold, enabling early identification of 3 out of 4 true readmissions.


