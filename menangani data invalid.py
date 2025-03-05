import pandas as pd
import numpy as np
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Eve', np.nan],
    'Age': [25, 30, 35, 25, np.nan, 150], # 150 dianggap sebagai nilai tidakvalid
    'City': ['New York', 'Los Angeles', 'Chicago', 'New York', 'Miami', 'Los Angeles']
}
df = pd.DataFrame(data)
# Identifikasi nilai tidak valid Data Setelah Menangani Nilai Tidak Valid: Name Age City 0 Alice 25.0 New York
print("Data Awal:")
print(df)
# Mengubah nilai 'Age' yang tidak valid menjadi NaN
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
# Mengisi nilai 'Age' yang hilang atau tidak valid dengan median
df['Age'] = df['Age'].fillna(df['Age'].median())
# Mengisi nilai 'Name' yang hilang dengan 'Unknown'
df['Name'] = df['Name'].fillna('Unknown')
print("\nData Setelah Menangani Nilai Tidak Valid:")
print(df)