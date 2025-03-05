# Contoh Confussion Matrix di NLP
# Impor library yang diperlukan
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
# Unduh dataset sentimen ulasan film dari NLTK
nltk.download('movie_reviews')
# Ambil ulasan dan label dari dataset
documents = [(list(movie_reviews.words(fileid)), category)
for category in movie_reviews.categories()
for fileid in movie_reviews.fileids(category)]
# Pisahkan teks ulasan dan label
texts = [' '.join(document) for document, category in documents]
labels = [category for document, category in documents]
# Ubah teks menjadi vektor fitur TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)
# Bagi dataset menjadi data pelatihan dan data pengujian
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2,
random_state=42)
# Inisialisasi dan latih model klasifikasi (misalnya, Linear SVM)
classifier = LinearSVC()
classifier.fit(X_train, y_train)
# Prediksi kelas pada data pengujian
y_pred = classifier.predict(X_test)
# Evaluasi model menggunakan confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
# Evaluasi model menggunakan classification report
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)