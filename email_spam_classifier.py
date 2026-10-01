"""
Email Spam Classifier
Tech Stack: Python, Pandas, Scikit-learn

Loads email text data, converts text to numbers (TF-IDF),
trains a Naive Bayes model, and classifies emails as spam or ham.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------
data = pd.read_csv("data/emails.csv")
print("First 5 rows of dataset:")
print(data.head())

# -------------------------------------------------
# 2. FEATURES AND TARGET
# -------------------------------------------------
X = data["text"]
y = data["label"]

# -------------------------------------------------
# 3. TRAIN-TEST SPLIT
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------------------------
# 4. CONVERT TEXT TO NUMBERS (TF-IDF)
# -------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------------------------
# 5. TRAIN MODEL
# -------------------------------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -------------------------------------------------
# 6. EVALUATE MODEL
# -------------------------------------------------
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------------------------
# 7. PREDICT ON NEW SAMPLE EMAILS
# -------------------------------------------------
sample_emails = [
    "Claim your free prize now click this link",
    "Let's meet for lunch tomorrow",
]

sample_vec = vectorizer.transform(sample_emails)
predictions = model.predict(sample_vec)

print("\n--- Sample Predictions ---")
for email, pred in zip(sample_emails, predictions):
    print(f"'{email}' -> {pred}")
