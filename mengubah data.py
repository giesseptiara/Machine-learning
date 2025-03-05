# Contoh: misalnya, kita memiliki dataset dengan kolom "City" yang berisi data kategorikal:
import pandas as pd
data = {'City': ['New York', 'Los Angeles', 'Chicago', 'New York', 'Miami']}
df = pd.DataFrame(data)
# One-Hot Encoding
df_one_hot = pd.get_dummies(df, columns=['City'])
print(df_one_hot)