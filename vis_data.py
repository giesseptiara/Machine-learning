# Distribusi Usia: Mari kita visualisasikan distribusi usia mahasiswa.

import matplotlib.pyplot as plt
import seaborn as sns

# Histogram untuk distribusi usia
plt.figure(figsize=(8, 6))
sns.histplot(df['Usia'], bins=5, kde=True)
plt.xlabel('Usia')
plt.ylabel('Frekuensi')
plt.show()