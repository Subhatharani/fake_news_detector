# fake_news_detector.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
# Make sure your CSV file has columns 'text' and 'label'
# label = 1 (real), 0 (fake)
df = pd.read_csv('train.csv')

# Step 2: Data preprocessing
df = df.fillna('')  # Fill missing values
X = df['text']      # Features (news content)
y = df['label']     # Labels (0 or 1)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 4: Text Vectorization using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test_tfidf)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Predict custom input
def predict_news(news):
    news_tfidf = vectorizer.transform([news])
    pred = model.predict(news_tfidf)[0]
    return "Real News 📰" if pred == 1 else "Fake News 🚫"

# Example usage
while True:
    print("\nEnter a news headline/content to predict (or type 'exit' to stop):")
    user_input = input("News: ")
    if user_input.lower() == 'exit':
        break
    print("Prediction:", predict_news(user_input))
