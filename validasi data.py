from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np
# Contoh dataset
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([2, 3, 4, 5, 6])
# Model
model = LinearRegression()
# K-Fold Cross-Validation
kf = KFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=kf)
print("K-Fold Cross-Validation Scores:", scores)
print("Mean Score:", np.mean(scores))