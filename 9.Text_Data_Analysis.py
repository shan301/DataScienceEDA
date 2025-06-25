# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 16:58:51 2025

@author: Shantanu
"""

"""Text Data Analysis
Text data analysis involves processing and analyzing unstructured text to extract insights, such as word frequencies, sentiment, or topics. This script covers essential techniques like text preprocessing, word frequency analysis, word clouds, and TF-IDF, using a sample dataset (text_data.csv).
"""

# Import required libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

"""1. Loading and Exploring Text Data
Load the text dataset and inspect its structure."""
# Load text data
df = pd.read_csv('data/text_data.csv')
print(df.head())  # Output: First 5 rows with ReviewID and Text columns
print(df.info())  # Output: Data types and non-null counts

"""2. Text Preprocessing
Preprocessing includes converting to lowercase, removing punctuation, tokenizing, removing stop words, and lemmatizing."""
def preprocess_text(text):
    """Preprocess a single text string.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing
df['Processed_Text'] = df['Text'].apply(preprocess_text)
print(df[['Text', 'Processed_Text']].head())  # Output: Original vs processed text

"""3. Word Frequency Analysis
Count the frequency of words to identify common terms."""
# Combine all processed text
all_text = ' '.join(df['Processed_Text'])
tokens = word_tokenize(all_text)
word_freq = Counter(tokens)
print("Top 5 words:", word_freq.most_common(5))  # Output: List of top 5 words with counts

"""4. Visualizing Word Frequencies
Plot a bar chart of the top 10 words."""
# Get top 10 words
top_words = dict(word_freq.most_common(10))
plt.figure(figsize=(10, 5))
plt.bar(top_words.keys(), top_words.values())
plt.title('Top 10 Word Frequencies')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()  # Output: Bar chart of top 10 words

"""5. Word Cloud
Generate a word cloud for visual representation of word frequencies."""
# Create word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Text Data')
plt.show()  # Output: Word cloud visualizing word frequencies

"""6. TF-IDF Analysis
TF-IDF (Term Frequency-Inverse Document Frequency) identifies important words in each document."""
# Compute TF-IDF
vectorizer = TfidfVectorizer(max_features=10)
tfidf_matrix = vectorizer.fit_transform(df['Processed_Text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
print(tfidf_df.head())  # Output: TF-IDF scores for top 10 terms

"""7. Sentiment Analysis (Basic)
Perform basic sentiment analysis using polarity scores."""
from textblob import TextBlob

# Calculate sentiment polarity
df['Sentiment'] = df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
print(df[['Text', 'Sentiment']].head())  # Output: Text with sentiment scores (-1 to 1)

"""Exercises
Practice the concepts learned with the following exercises using text_data.csv."""

"""Exercise 1: Load and Inspect
Load text_data.csv and display the first 10 rows."""
df_ex1 = pd.read_csv('data/text_data.csv')
print(df_ex1.head(10))  # Output: First 10 rows of text_data.csv

"""Exercise 2: Count Non-Null Texts
Count the number of non-null entries in the Text column."""
print(df['Text'].notnull().sum())  # Output: Number of non-null text entries

"""Exercise 3: Basic Preprocessing
Convert the Text column to lowercase and print the first 5 rows."""
df['Text_Lower'] = df['Text'].str.lower()
print(df[['Text', 'Text_Lower']].head())  # Output: Original vs lowercase text

"""Exercise 4: Remove Punctuation
Remove punctuation from the Text column and print the first 5 rows."""
df['Text_NoPunct'] = df['Text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
print(df[['Text', 'Text_NoPunct']].head())  # Output: Original vs punctuation-free text

"""Exercise 5: Tokenization
Tokenize the Text column and print the tokens for the first row."""
df['Tokens'] = df['Text'].apply(word_tokenize)
print(df['Tokens'].iloc[0])  # Output: List of tokens for first row

"""Exercise 6: Remove Stop Words
Remove stop words from the tokenized Text column and print the first 5 rows."""
stop_words = set(stopwords.words('english'))
df['Tokens_NoStop'] = df['Tokens'].apply(lambda x: [word for word in x if word not in stop_words])
print(df[['Tokens', 'Tokens_NoStop']].head())  # Output: Tokens vs stop-word-free tokens

"""Exercise 7: Lemmatization
Lemmatize the tokens (after removing stop words) and print the first 5 rows."""
lemmatizer = WordNetLemmatizer()
df['Tokens_Lemmatized'] = df['Tokens_NoStop'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
print(df[['Tokens_NoStop', 'Tokens_Lemmatized']].head())  # Output: Stop-word-free vs lemmatized tokens

"""Exercise 8: Word Frequency
Calculate the frequency of all words in the Processed_Text column."""
all_tokens = word_tokenize(' '.join(df['Processed_Text']))
freq_ex8 = Counter(all_tokens)
print("Top 10 words:", freq_ex8.most_common(10))  # Output: Top 10 words with counts

"""Exercise 9: Bar Plot of Top Words
Create a bar plot of the top 5 words by frequency."""
top_words_ex9 = dict(freq_ex8.most_common(5))
plt.figure(figsize=(8, 4))
plt.bar(top_words_ex9.keys(), top_words_ex9.values())
plt.title('Top 5 Word Frequencies')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()  # Output: Bar chart of top 5 words

"""Exercise 10: Word Cloud
Generate a word cloud from the Processed_Text column."""
wordcloud_ex10 = WordCloud(width=600, height=300, background_color='white').generate(' '.join(df['Processed_Text']))
plt.figure(figsize=(8, 4))
plt.imshow(wordcloud_ex10, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud')
plt.show()  # Output: Word cloud of processed text

"""Exercise 11: TF-IDF for Top 5 Terms
Compute TF-IDF scores for the top 5 terms and display the result."""
vectorizer_ex11 = TfidfVectorizer(max_features=5)
tfidf_matrix_ex11 = vectorizer_ex11.fit_transform(df['Processed_Text'])
tfidf_df_ex11 = pd.DataFrame(tfidf_matrix_ex11.toarray(), columns=vectorizer_ex11.get_feature_names_out())
print(tfidf_df_ex11.head())  # Output: TF-IDF scores for top 5 terms

"""Exercise 12: Sentiment Analysis
Calculate sentiment polarity for the Text column and print the top 5 positive reviews."""
df['Sentiment_Ex12'] = df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
print(df[['Text', 'Sentiment_Ex12']].nlargest(5, 'Sentiment_Ex12'))  # Output: Top 5 positive reviews

"""Exercise 13: Average Word Length
Calculate the average word length in the Processed_Text column."""
df['Word_Lengths'] = df['Processed_Text'].apply(lambda x: [len(word) for word in word_tokenize(x)])
avg_word_length = df['Word_Lengths'].apply(lambda x: sum(x) / len(x) if x else 0).mean()
print(f"Average word length: {avg_word_length:.2f}")  # Output: Average word length (e.g., 4.50)

"""Exercise 14: Unique Words
Count the number of unique words in the Processed_Text column."""
unique_words = len(set(word_tokenize(' '.join(df['Processed_Text']))))
print(f"Number of unique words: {unique_words}")  # Output: Number of unique words

"""Exercise 15: Filter Long Reviews
Filter reviews with more than 50 words and print the first 5."""
df['Word_Count'] = df['Text'].apply(lambda x: len(word_tokenize(x)))
long_reviews = df[df['Word_Count'] > 50]
print(long_reviews[['Text', 'Word_Count']].head())  # Output: First 5 reviews with >50 words

"""Notes
- Ensure text_data.csv has ReviewID and Text columns.
- NLTK requires downloading 'punkt', 'stopwords', and 'wordnet' for tokenization, stop word removal, and lemmatization.
- TextBlob is used for sentiment analysis; install it with `pip install textblob`.
- Adjust max_features in TfidfVectorizer based on dataset size.
- For advanced analysis, consider topic modeling (e.g., LDA) or named entity recognition (not covered here).
"""

if __name__ == "__main__":
    # Run all sections when script is executed
    pass