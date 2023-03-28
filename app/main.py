import nltk
import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
news_df = pd.read_csv('news_dataset.csv')

# Preprocess text data
def preprocess_text(text):
    # Tokenize text into words
    words = nltk.word_tokenize(text.lower())
    # Remove stop words
    stopwords = nltk.corpus.stopwords.words('english')
    words = [word for word in words if word not in stopwords]
    # Remove non-alphabetic characters
    words = [word for word in words if word.isalpha()]
    # Lemmatize words
    lemmatizer = nltk.stem.WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join words into text
    return ' '.join(words)

# Apply text preprocessing to news_df
news_df['text'] = news_df['text'].apply(preprocess_text)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(news_df['text'], news_df['category'], test_size=0.2, random_state=42)

# Train and evaluate model
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Serialize and save model
with open('news_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)
