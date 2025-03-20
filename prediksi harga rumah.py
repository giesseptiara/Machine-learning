# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load Dataset (contoh dataset)
df = pd.read_csv("house_prices.csv")  # Gantilah dengan dataset yang sesuai
print(df.head())

# 2. Preprocessing Data
df = df[['Size', 'Location', 'YearBuilt', 'Price']]  # Pilih fitur yang relevan
df = pd.get_dummies(df, columns=['Location'], drop_first=True)  # One-hot encoding lokasi

# 3. Split Data
X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Random Search untuk Random Forest
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}
rf_model = RandomizedSearchCV(RandomForestRegressor(random_state=42), rf_params, n_iter=10, cv=5, scoring='neg_mean_absolute_error')
rf_model.fit(X_train, y_train)
print("Best Random Forest Parameters:", rf_model.best_params_)

# 5. Random Search untuk Gradient Boosting
gb_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10]
}
gb_model = RandomizedSearchCV(GradientBoostingRegressor(random_state=42), gb_params, n_iter=10, cv=5, scoring='neg_mean_absolute_error')
gb_model.fit(X_train, y_train)
print("Best Gradient Boosting Parameters:", gb_model.best_params_)

# 6. Evaluasi Model
best_rf = rf_model.best_estimator_
best_gb = gb_model.best_estimator_

y_pred_rf = best_rf.predict(X_test)
y_pred_gb = best_gb.predict(X_test)

print("\nRandom Forest MAE:", mean_absolute_error(y_test, y_pred_rf))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

print("\nGradient Boosting MAE:", mean_absolute_error(y_test, y_pred_gb))
print("Gradient Boosting RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_gb)))

# 7. Visualisasi Hasil
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_rf, alpha=0.5, label="Random Forest")
plt.scatter(y_test, y_pred_gb, alpha=0.5, label="Gradient Boosting", color="red")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='black')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.legend()
plt.title("Prediksi Harga Rumah")
plt.show()
