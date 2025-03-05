from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
# Bagi dataset menjadi data pelatihan dan data pengujian
X_train, X_test, y_train, y_test = train_test_split(X,
random_state=42)
y, test_size=0.2,
# Inisialisasi dan latih model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# Prediksi dengan data pengujian
y_pred = model.predict(X_test)
# Evaluasi kinerja model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)