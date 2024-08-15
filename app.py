import flask
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer

app = flask.Flask(__name__, template_folder='templates')

# Load the Career Recommendation Dataset
df = pd.read_csv('./career_recommendation_dataset.csv')

# Preprocess the data (combine relevant columns into a single text column)
df['tags'] = df['Skills'] + ' ' + df['Job_Title'] + ' ' + df['Description']

# Convert tags to lowercase
df['tags'] = df['tags'].str.lower()

# Train a Word2Vec model on the tags
tags = [tag.split() for tag in df['tags']]
word2vec_model = Word2Vec(sentences=tags, vector_size=100, window=5, min_count=1, workers=4)

# Create embeddings for each career based on the average of word embeddings
def get_embeddings(text):
    words = text.split()
    embeddings = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(100)

df['embeddings'] = df['tags'].apply(get_embeddings)

# Hybrid model: K-NN + Cosine Similarity
def get_recommendations(career_title, k=10):
    # Find the index of the career title
    idx = df.index[df['Job_Title'].str.lower() == career_title.lower()].tolist()
    if not idx:
        return None
    idx = idx[0]
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity([df['embeddings'].iloc[idx]], list(df['embeddings']))
    
    # Combine K-NN and cosine similarity for hybrid recommendation
    knn_model = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn_model.fit(list(df['embeddings']))
    
    knn_indices = knn_model.kneighbors([df['embeddings'].iloc[idx]], return_distance=False)[0]
    
    recommendations = [(df['Job_Title'].iloc[i], cosine_sim[0][i]) for i in knn_indices]
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    
    return recommendations[:k]

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('index.html')

    if flask.request.method == 'POST':
        career_name = flask.request.form['career_name']
        career_name = career_name.title()
        
        result_final = get_recommendations(career_name)
        if result_final is None:
            return flask.render_template('negative.html', name=career_name)
        else:
            titles, scores = zip(*result_final)
            return flask.render_template('positive.html', career_titles=titles, similarity_scores=scores, search_name=career_name)

if __name__ == '__main__':
    app.run()
