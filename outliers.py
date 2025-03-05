# Langkah 1: Membuat DataFrame
import pandas as pd
import numpy as np
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Eve', np.nan],
    'Age': [25, 30, 35, 25, np.nan, 50],
    'City': ['New York', 'Los Angeles', 'Chicago', 'New York', 'Miami', 'Los Angeles']
}
df = pd.DataFrame(data)
# Langkah 2: Identifikasi Outliers
# Outliers dapat diidentifikasi menggunakan berbagai metode, salah satu yang umum adalah menggunakan Z-score atau IQR (Interquartile Range).
# Di sini, kita akan menggunakan IQR untuk mendeteksi outliers.
# Menghitung IQR
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
# Menentukan batas bawah dan atas untuk outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Menandai outliers
outliers = df[(df['Age'] < lower_bound) | (df['Age'] > upper_bound)]
print("Outliers:\n", outliers)
# Langkah 3: Menangani Outliers
# # Menghapus outliers
df_no_outliers = df[(df['Age'] >= lower_bound) & (df['Age'] <= upper_bound)]
# Mengganti outliers dengan nilai median
median_age = df['Age'].median()
df['Age'] = np.where((df['Age'] < lower_bound) | (df['Age'] > upper_bound),
median_age, df['Age'])
# Mengisi missing values dengan median
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Name'].fillna('Unknown', inplace=True)

