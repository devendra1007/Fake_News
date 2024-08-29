import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Create a results folder if it doesn't exist
import os
if not os.path.exists('../results'):
    os.makedirs('../results')

# Read the data
df = pd.read_csv('../Data/news.csv')

# Get the labels
labels = df.label

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.1, random_state=7)

# Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)

# Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)

# Build confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

# Save accuracy and confusion matrix to files
with open('../results/accuracy.txt', 'w') as f:
    f.write(f'Accuracy: {round(score * 100, 2)}%')

# Plot and save confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('../results/confusion_matrix.png')
plt.close()

# Generate and save word cloud
text = ' '.join(df['text'])
wordcloud = WordCloud(stopwords='english', background_color='white').generate(text)

plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of News Text')
plt.savefig('../results/word_cloud.png')
plt.close()

# Optional: Additional Analysis
features = tfidf_vectorizer.get_feature_names_out()
coef = pac.coef_[0]  # No need to call toarray() here
feature_importance = sorted(zip(features, coef), key=lambda x: x[1], reverse=True)
top_features = feature_importance[:20]

# Save top features to file
with open('../results/top_features.txt', 'w') as f:
    f.write('Top 20 Important Features:\n')
    for feature, importance in top_features:
        f.write(f'{feature}: {importance}\n')
