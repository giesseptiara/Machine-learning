import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('students.csv')

# Hitung rata-rata nilai per mata pelajaran
mean_scores = data[['Matematika', 'IPA', 'Bahasa Inggris']].mean()

# Visualisasi rata-rata nilai
plt.figure(figsize=(8, 5))
mean_scores.plot(kind='bar', color=['blue', 'green', 'red'])
plt.title('Rata-rata Nilai Siswa')
plt.xlabel('Mata Pelajaran')
plt.ylabel('Rata-rata Nilai')
plt.grid(axis='y')
plt.show()
