import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Contoh dataset
data = {
    'Age': [25, 30, 35, 40, 45],
    'Salary': [50000, 60000, 70000, 80000, 90000]
}
df = pd.DataFrame(data)
# Min-Max Scaling
min_max_scaler = MinMaxScaler()
df_min_max_scaled = pd.DataFrame(min_max_scaler.fit_transform(df),
columns=df.columns)
# Z-Score Normalization
standard_scaler = StandardScaler()
df_standard_scaled = pd.DataFrame(standard_scaler.fit_transform(df),
columns=df.columns)
print("Data Asli:")
print(df)
print("\nMin-Max Scaled Data:")
print(df_min_max_scaled)
print("\nStandardized Data:")
print(df_standard_scaled)