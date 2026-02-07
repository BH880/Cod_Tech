import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

data = {
    'text': [
        'Win money now!!!',
        'Hello friend, how are you?',
        'Get cheap meds now',
        'Important project meeting tomorrow',
        'You have won a lottery prize',
        'Let’s catch up soon'
    ],
    'label': [1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df['text']
y = df['label']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# stratify to keep class balance in test set
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.3, random_state=42, stratify=y
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

new_email = ["Congratulations, you won a free vacation!"]
new_vec = vectorizer.transform(new_email)
prediction = model.predict(new_vec)
print("Spam" if prediction[0] == 1 else "Not Spam")
