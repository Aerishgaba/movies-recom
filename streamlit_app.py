import pandas as pd

# Load the dataset
data = pd.read_csv('/workspaces/movies-recom/data/imdb_movies.csv')

# View the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Check data types
print(data.dtypes)


data = data.dropna(subset=['overview'])

import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stop words
    return text

# Apply to the overview column
data['overview_cleaned'] = data['overview'].apply(clean_text)


data['features'] = data['overview_cleaned'] + ' ' + data['genre'].fillna('')

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the vectorizer
tfidf = TfidfVectorizer(min_df=3, ngram_range=(1, 2))

# Fit and transform the features
tfidf_matrix = tfidf.fit_transform(data['features'])


from sklearn.metrics.pairwise import cosine_similarity

# Compute the similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_movies(movie_title, data, similarity_matrix, top_n=5):
    # Find the movie index
    movie_idx = data[data['names'] == movie_title].index
    if len(movie_idx) == 0:
        return "Movie not found!"
    
    movie_idx = movie_idx[0]
    
    # Get similarity scores for this movie
    sim_scores = list(enumerate(similarity_matrix[movie_idx]))
    
    # Sort by similarity (highest first) and exclude the movie itself
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    # Get movie titles
    movie_indices = [i[0] for i in sim_scores]
    return data['names'].iloc[movie_indices].tolist()



import streamlit as st

st.title("Movie Recommendation System")
movie_title = st.text_input("Enter a movie title:")
if movie_title:
    recommendations = recommend_movies(movie_title, data, similarity_matrix)
    st.write("Recommended Movies:")
    for movie in recommendations:
        st.write(f"- {movie}")