import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re

import normalizers

#
# We use a pre-scraped dataset to generate feature vectors by concatenating the 'ingredients' and 'directions'
# for rows that have matching URLs with the annotated data.
#
def load_dataset(annotated_path, scraped_path):
    
    def combine_columns(row):
        combined_text = f"{row['ingredients']} {row['directions']}"
        return combined_text.lower()

    annotated_data = pd.read_csv(annotated_path)
    scraped_data = pd.read_csv(scraped_path)

    matched_data = annotated_data.merge(scraped_data, left_on='url', right_on='url', how='inner')
    matched_data = matched_data.drop_duplicates(subset='url')

    if 'ingredients' not in matched_data.columns or 'directions' not in matched_data.columns:
        raise ValueError("Columns 'ingredients' and 'directions' must be present in the dataset.")

    x = [normalize(combine_columns(row)) for _, row in matched_data.iterrows()]
    y = matched_data.iloc[:, 1].values

    return x, y

#
# We apply the several cleaning and normalization steps to the data set. 
#
def normalize(feature):

    feature = feature.strip()
    feature = re.sub(r'\n{2,}.*$', '', feature, flags=re.DOTALL)
    
    # Tokenize
    sentences = nltk.sent_tokenize(feature)
    words = [nltk.word_tokenize(sentence) for sentence in sentences]

    # Remove unwanted characters
    words = [[word for word in sentence if word not in string.punctuation] for sentence in words]
    words = [[re.sub(r'[^a-zA-Z0-9]', '', word) for word in sentence] for sentence in words]
    words = [normalizers.remove_numbers(word) for word in words]

    return np.concatenate(words).tolist()

from sklearn.feature_extraction.text import CountVectorizer

def preprocess_bow(features):
    features_text = [' '.join(feature) for feature in features]

    vectorizer = CountVectorizer()

    bow_matrix = vectorizer.fit_transform(features_text)
    
    return bow_matrix

def preprocess_tfidf(features):
    features_text = [' '.join(feature) for feature in features]

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(features_text)
    
    return tfidf_matrix
