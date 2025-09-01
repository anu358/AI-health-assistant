import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Load dataset
df = pd.read_csv("symptom_disease.csv")

# Combine symptoms into one text field
df["combined"] = df.iloc[:, :-1].apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)

X = df["combined"]
y = df["disease"]

# Vectorize text
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model + vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved! Accuracy:", model.score(X_test, y_test))
