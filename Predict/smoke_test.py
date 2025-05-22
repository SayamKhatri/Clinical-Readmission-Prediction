import pandas as pd
from predict import predict  

df_raw = pd.read_csv('/Users/samkhatri/Desktop/Data Science Projects/Clinical-Readmission-Prediction/Data/diabetic_data.csv').head(5)

out = predict(df_raw)

print(out)
