# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset (contoh dataset)
df = pd.read_csv("spam.csv", encoding="latin-1")  # Gantilah dengan dataset Anda
df = df[['v1', 'v2']]  # Menggunakan kolom kategori (v1) dan teks (v2)
df.columns = ['label', 'text']

# 2. Encoding Label (Spam=1, Non-Spam=0)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# 4. TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Grid Search untuk Naive Bayes
nb_params = {'alpha': [0.1, 0.5, 1.0, 2.0]}
nb_model = GridSearchCV(MultinomialNB(), nb_params, cv=5, scoring='accuracy')
nb_model.fit(X_train_vec, y_train)
print("Best Naive Bayes Parameters:", nb_model.best_params_)

# 6. Grid Search untuk SVM
svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svm_model = GridSearchCV(SVC(), svm_params, cv=5, scoring='accuracy')
svm_model.fit(X_train_vec, y_train)
print("Best SVM Parameters:", svm_model.best_params_)

# 7. Evaluasi Model
best_nb = nb_model.best_estimator_
best_svm = svm_model.best_estimator_

y_pred_nb = best_nb.predict(X_test_vec)
y_pred_svm = best_svm.predict(X_test_vec)

print("\nNaive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nSVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# 8. Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Naive Bayes")
plt.show()
