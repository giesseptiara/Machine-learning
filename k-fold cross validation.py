# Implementasi K-Fold Cross-Validation di Python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
# Model
model = LogisticRegression(max_iter=1000)
# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42) # Misalnya, menggunakan 5 folds dengan shuffle dan seed random 42
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print("K-Fold Cross-Validation Scores:", scores)
print("Mean Score:", scores.mean())