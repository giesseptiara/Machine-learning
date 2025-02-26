from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Memuat dataset
df = pd.read_csv('students.csv')

# Menampilkan 5 baris pertama dataset
print(df.head())

# Memeriksa ukuran dataset
print(f"Ukuran dataset: {df.shape}")

# Memeriksa tipe data setiap kolom
print(df.dtypes)

# Memeriksa nilai yang hilang
print(df.isnull().sum())

# Deskripsi statistik dasar
print(df.describe(include='all'))

# Distribusi Usia: Mari kita visualisasikan distribusi usia mahasiswa.


# Histogram untuk distribusi usia
plt.figure(figsize=(8, 6))
sns.histplot(df['Usia'], bins=5, kde=True)
plt.xlabel('Usia')
plt.ylabel('Frekuensi')
plt.show()

# Distribusi Nilai: Visualisasikan distribusi nilai untuk setiap mata pelajaran.
# Boxplot untuk nilai setiap mata pelajaran
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['Matematika', 'IPA', 'Bahasa Inggris']])
plt.title('Distribusi Nilai Mahasiswa')
plt.xlabel('Mata Pelajaran')
plt.ylabel('Nilai')
plt.show()

# Perbandingan Nilai Berdasarkan Jenis Kelamin dan Jurusan
# Visualisasikan perbandingan nilai berdasarkan jenis kelamin dan jurusan.
# Scatter plot untuk perbandingan nilai berdasarkan jenis kelamin
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Matematika', y='IPA', hue='Jenis Kelamin', style='Jurusan',
data=df, s=100)
plt.title('Perbandingan Nilai Matematika dan IPA Berdasarkan Jenis Kelamin dan Jurusan')
plt.xlabel('Nilai Matematika')
plt.ylabel('Nilai IPA')
plt.legend(title='Jenis Kelamin & Jurusan')
plt.show()
# Boxplot untuk perbandingan nilai berdasarkan jurusan
plt.figure(figsize=(10, 6))
sns.boxplot(x='Jurusan', y='Matematika', hue='Jenis Kelamin', data=df)
plt.title('Perbandingan Nilai Matematika Berdasarkan Jurusan dan Jenis Kelamin')
plt.xlabel('Jurusan')
plt.ylabel('Nilai Matematika')
plt.show()

# !pip install -U scikit-learn
from sklearn.datasets import load_iris
import pandas as pd
# Memuat dataset Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
# Menampilkan 5 baris pertama dataset
df.head()# Memeriksa ukuran dataset
print(df.shape)
# Memeriksa tipe data setiap kolom
print(df.dtypes)
# Memeriksa nilai yang hilang
print(df.isnull().sum())
# Deskripsi statistik dasar
print(df.describe())