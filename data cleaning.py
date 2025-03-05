# Contoh Data Cleaning dengan Python
import pandas as pd
import numpy as np

# Membuat dataframe contoh
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Eve', np.nan],
    'Age': [25, 30, 35, 25, np.nan, 50],
    'City': ['New York', 'Los Angeles', 'Chicago', 'New York', 'Miami', 'Los Angeles']
}
df = pd.DataFrame(data)

# Menampilkan data awal
print("Data Awal:")
print(df)

# Menghapus duplikat
df = df.drop_duplicates()

# Menangani missing values dengan mengisi nilai median untuk kolom numerik
df['Age'] = df['Age'].fillna(df['Age'].median())

# Menghapus baris yang mengandung missing values di kolom 'Name'
df = df.dropna(subset=['Name'])

# Menampilkan data setelah cleaning
print("\nData Setelah Cleaning:")
df

# Identifikasi Missing Values
df = pd.read_csv('students.csv')
print(df.isnull().sum())

# Mengisi Missing Values df['Age'].fillna(df['Age'].mean())
# Mean df['Age'].fillna(df['Age'].median(), inplace=True) 
# Median df['Age'].fillna(df['Age'].mode()[0], inplace=True)
# Mode

# Forward/Backward Fill
df['Age'].fillna(method='ffill', inplace=True) # Forward fill
df['Age'].fillna(method='bfill', inplace=True) # Backward fill

# Dropping Missing Values
df.dropna(subset=['Age'], inplace=True)

# Identifikasi duplikat
print(df.duplicated().sum())
# Menghapus duplikat
df.drop_duplicates(inplace=True)

