import pandas as pd
import numpy as np

def icd9_chapter(code):
    if pd.isna(code) or code == '?' or str(code).strip() == '':
        return None
    s = str(code)
    if s.startswith('E'): return 'External'
    if s.startswith('V'): return 'Supplementary'
    try:
        n = float(s)
    except:
        return None
    if   1 <= n <= 139:  return 'Infectious'
    elif 140 <= n <= 239: return 'Neoplasm'
    elif 240 <= n <= 279: return 'Endocrine'
    elif 280 <= n <= 289: return 'Blood'
    elif 290 <= n <= 319: return 'Mental'
    elif 320 <= n <= 389: return 'NervousSystem'
    elif 390 <= n <= 459: return 'Circulatory'
    elif 460 <= n <= 519: return 'Respiratory'
    elif 520 <= n <= 579: return 'Digestive'
    elif 580 <= n <= 629: return 'Genitourinary'
    elif 630 <= n <= 679: return 'Pregnancy'
    elif 680 <= n <= 709: return 'Skin'
    elif 710 <= n <= 739: return 'Musculoskeletal'
    elif 740 <= n <= 759: return 'Congenital'
    elif 760 <= n <= 779: return 'Perinatal'
    elif 780 <= n <= 799: return 'Symptoms'
    elif 800 <= n <= 999: return 'Injury'
    else: return None

def charlson_score(codes):

    charlson_map = {}
    def add(codes, w):
        for c in codes:
            charlson_map[c] = w
    add(['410','412'],1)
    add(['39891','40201','40211','40291','40401','40403','40411','40413',
         '40491','40493','428'],1)
    add(['0930','4373','440','441','442','4431','4432','4438','4439',
         '4471','5571','5579','V434'],1)
    add(['430','431','432','433','434','435','436','4373'],1)
    add(['290'],1)
    add(['4168','4169','490','491','492','493','494','495','496',
         '500','501','502','503','504','505'],1)
    add(['4465','7100','7101','7104','7109','7140','7141','7142',
         '7148','725'],1)
    add(['531','532','533','534'],1)
    add(['5712','5714','5715','5716'],1)
    add([f'250{str(i).zfill(2)}' for i in range(0,100)],1)
    add(['342','343','344'],2)
    add(['40301','40311','40391','40402','40403','40412','40413',
         '40492','40493','585','586','5888','5889'],2)
    add([str(i) for i in range(140,209)],2)
    add(['5722','5723','5724','5728'],3)
    add(['196','197','198','199'],6)
    add(['042','043','044'],6)
    total = 0
    for code in codes:
        s = str(code)

        for L in (4,3):
            key = s[:L]
            if key in charlson_map:
                total += charlson_map[key]
                break
    return total


def build_features(df_raw):
    df = df_raw.copy()


    drop1 = ['nateglinide','chlorpropamide','acetohexamide','tolbutamide',
             'acarbose','miglitol','troglitazone','tolazamide','examide',
             'citoglipton','glyburide-metformin','glipizide-metformin',
             'glimepiride-pioglitazone','metformin-rosiglitazone',
             'metformin-pioglitazone']
    df.drop(columns=drop1, errors='ignore', inplace=True)


    df['readmitted_binary'] = df['readmitted'].map({'<30':1}).fillna(0).astype(int)


    df['insulin_prescribed'] = (df['insulin'] != 'No').astype(int)
    df['insulin_up']         = (df['insulin']=='Up').astype(int)
    df['insulin_down']       = (df['insulin']=='Down').astype(int)
    df['insulin_steady']     = (df['insulin']=='Steady').astype(int)

 
    oral = ['metformin','repaglinide','glimepiride','glipizide',
            'glyburide','pioglitazone','rosiglitazone']
    
    df['num_oral_meds_prescribed'] = df[oral].ne('No').sum(axis=1)
    df['num_oral_meds_up']        = df[oral].eq('Up').sum(axis=1)
    df['num_oral_meds_down']      = df[oral].eq('Down').sum(axis=1)
    df['oral_med_changed']        = ((df['num_oral_meds_up']+
                                      df['num_oral_meds_down'])>0).astype(int)
    df['oral_med_change_ratio']   = (
        df['num_oral_meds_up']+df['num_oral_meds_down']
    ).div(df['num_oral_meds_prescribed'].replace(0,1))


    df.drop(columns=oral + ['insulin','change','diabetesMed'],
            errors='ignore', inplace=True)


    glu_map = {'None':0,'Norm':1,'>200':2,'>300':3}
    a1c_map= {'None':0,'Norm':1,'>7':2,'>8':3}
    df['max_glu_serum_ord'] = df['max_glu_serum'].map(glu_map).fillna(0).astype(int)
    df['A1Cresult_ord']    = df['A1Cresult'].map(a1c_map).fillna(0).astype(int)

    keep_at=[1,2,3]
    df['admission_type_grp'] = (
        df['admission_type_id']
          .where(df['admission_type_id'].isin(keep_at),'Other')
          .map({1:'Emergency',2:'Urgent',3:'Elective','Other':'Other'})
    )

    keep_as=[7,1,17,4,6,2]
    src_map={7:'ER',1:'PhysicianReferral',17:'Unknown',
             4:'HospitalTransfer',6:'FacilityTransfer',2:'ClinicReferral'}
    df['admission_source_grp'] = (
        df['admission_source_id']
          .where(df['admission_source_id'].isin(keep_as),'Other')
          .map(lambda x: src_map.get(x,'Other'))
    )

    keep_dd=[1,3,6,18,2,22,11,5]
    dd_map={1:'Home',3:'SNF',6:'HomeHealth',18:'HospiceHome',
            2:'OtherHospital',22:'HospiceFacility',11:'RehabFacility',
            5:'OtherCare'}
    df['discharge_disp_grp'] = (
        df['discharge_disposition_id']
          .where(df['discharge_disposition_id'].isin(keep_dd),'Other')
          .map(lambda x: dd_map.get(x,'Other'))
    )

    
    df['race']   = df['race'].replace('?', 'Unknown')
    df['gender'] = df['gender'].replace('Unknown/Invalid','Unknown')

    def age_grp(x):
        n = int(x.strip('[]()').split('-')[0])
        return ('under25' if n<25 else '25to49' if n<50
                else '50to74' if n<75 else '75plus')
    
    df['age_grp']= df['age'].apply(age_grp)

 
    df['payer_grp']   = df['payer_code'].replace('?', 'Unknown')
    df['medspec_grp'] = df['medical_specialty'].replace('?', 'Unknown')
 

    def collapse(series, pct=0.01):
        fr = series.value_counts(normalize=True)
        keep = fr[fr>=pct].index
        return series.where(series.isin(keep),'Other')
    df['payer_grp']   = collapse(df['payer_grp'])
    df['medspec_grp'] = collapse(df['medspec_grp'])
    

    to_ohe = ['admission_type_grp','admission_source_grp','discharge_disp_grp',
              'race','gender','age_grp','payer_grp','medspec_grp']
    df = pd.get_dummies(df, columns=to_ohe, prefix=['admType','admSrc','disp',
                                                   'race','gender','age',
                                                   'payer','medspec'],
                        drop_first=False)

   
    base = df['readmitted_binary'].mean()
    lifts = (df.groupby('diag_1')['readmitted_binary'].mean() - base).abs()
    top10 = lifts.nlargest(10).index.tolist()
    df['diag1_grp'] = df['diag_1'].where(df['diag_1'].isin(top10),'Other')
    df = pd.get_dummies(df, columns=['diag1_grp'], prefix='diag1',
                        drop_first=False)


    df['age_insulin_risk']    = df.get('age_75plus',0)*df['insulin_prescribed']
    df['unstable_patient']    = ((df['number_inpatient']>0)&
                                 (df['oral_med_changed']==1)).astype(int)
    df['emergency_med_change']= df.get('admType_Emergency',0)*df['oral_med_changed']
    m, l = df['time_in_hospital'].median(), df['num_procedures'].median()
    df['complex_stay']        = ((df['time_in_hospital']>m)&
                                 (df['num_procedures']>l)).astype(int)


    df['charlson_index'] = df[['diag_1','diag_2','diag_3']].apply(
        lambda row: charlson_score(row), axis=1)
    df['total_admissions']= df.groupby('patient_nbr')['encounter_id'].transform('count')
    df['prior_admissions']= df['total_admissions'] - 1

  
    num_feats = ['time_in_hospital','num_lab_procedures','num_procedures',
                 'num_medications','number_outpatient','number_emergency',
                 'number_inpatient','number_diagnoses','insulin_prescribed',
                 'insulin_up','insulin_down','num_oral_meds_prescribed',
                 'num_oral_meds_up','num_oral_meds_down','oral_med_changed',
                 'oral_med_change_ratio','max_glu_serum_ord','A1Cresult_ord']
    admType_feats   = [c for c in df if c.startswith('admType_')]
    admSrc_feats    = [c for c in df if c.startswith('admSrc_')]
    disp_feats      = [c for c in df if c.startswith('disp_')]
    race_feats      = [c for c in df if c.startswith('race_')]
    gender_feats    = [c for c in df if c.startswith('gender_')]
    age_feats       = [c for c in df if c.startswith('age_')]
    payer_feats     = [c for c in df if c.startswith('payer_')]
    medspec_feats   = [c for c in df if c.startswith('medspec_')]
    diag1_feats     = [c for c in df if c.startswith('diag1_')]
    inter_feats     = ['age_insulin_risk','unstable_patient',
                       'emergency_med_change','complex_stay',
                       'charlson_index','prior_admissions']

    feature_cols = (num_feats + admType_feats + admSrc_feats + disp_feats +
                    race_feats + gender_feats + age_feats +
                    payer_feats + medspec_feats + diag1_feats +
                    inter_feats)

 
    X = (
        df[feature_cols]
        .apply(pd.to_numeric, errors='coerce')  
        .fillna(0)                               
        .astype(float)
    )
    y = df['readmitted_binary'].astype(int)

    return X, y, feature_cols
