import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Contoh dataset
data = np.array([[1.0, 2.0, 3.0],
[4.0, 5.0, 6.0],
[7.0, 8.0, 9.0]])
# 1. Min-Max Scaling (Normalization)
scaler_minmax = MinMaxScaler()
data_minmax_scaled = scaler_minmax.fit_transform(data)
print("Min-Max Scaled Data:")
print(data_minmax_scaled)
print()
# 2. Z-Score Normalization (Standardization)
scaler_standard = StandardScaler()
data_standard_scaled = scaler_standard.fit_transform(data)
print("Z-Score Standardized Data:")
print(data_standard_scaled)