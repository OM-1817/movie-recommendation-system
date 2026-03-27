from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# =========================
# LOAD DATA (runs once)
# =========================

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# preprocess genres
movies['genres'] = movies['genres'].str.replace('|', ' ')

# content-based matrix
cv = CountVectorizer(stop_words='english')
matrix = cv.fit_transform(movies['genres'])
similarity = cosine_similarity(matrix)

# user-item matrix
user_movie_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

# =========================
# HELPER FUNCTIONS
# =========================

def get_movie_title(movie_id):
    return movies[movies['movieId'] == movie_id]['title'].values[0]


def get_similar_users(user_id, n=5):
    from sklearn.metrics.pairwise import cosine_similarity
    user_similarity = cosine_similarity(user_movie_matrix)
    sim_scores = user_similarity[user_id - 1]
    similar_users = np.argsort(sim_scores)[::-1][1:n+1]
    return similar_users


def content_recommend_with_scores(movie_name, n=20):
    try:
        movie_index = movies[movies['title'] == movie_name].index[0]
    except IndexError:
        raise HTTPException(status_code=404, detail="Movie not found")

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:n+1]

    content_scores = {}

    for idx, score in movie_list:
        movie_id = movies.iloc[idx].movieId
        content_scores[movie_id] = score

    return content_scores


def get_collaborative_scores(user_id, n_similar=5):
    similar_users = get_similar_users(user_id, n_similar)

    collab_scores = {}

    for sim_user in similar_users:
        sim_user_id = sim_user + 1
        sim_user_movies = user_movie_matrix.loc[sim_user_id]

        for movie_id, rating in sim_user_movies.items():
            if rating > 3:
                collab_scores[movie_id] = collab_scores.get(movie_id, 0) + rating

    # normalize
    max_possible = n_similar * 5
    for movie_id in collab_scores:
        collab_scores[movie_id] /= max_possible

    return collab_scores


def hybrid_recommend(user_id, movie_name, n=5, alpha=0.6, beta=0.4):
    content_scores = content_recommend_with_scores(movie_name)
    collab_scores = get_collaborative_scores(user_id)

    # watched movies
    watched = user_movie_matrix.loc[user_id]
    watched_movies = set(watched[watched > 0].index)

    hybrid_scores = {}

    all_movies = set(content_scores.keys()).union(set(collab_scores.keys()))

    for movie_id in all_movies:
        if movie_id in watched_movies:
            continue

        content_score = content_scores.get(movie_id, 0)
        collab_score = collab_scores.get(movie_id, 0)

        hybrid_scores[movie_id] = alpha * content_score + beta * collab_score

    sorted_movies = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

    recommendations = []

    for movie_id, _ in sorted_movies[:n]:
        recommendations.append(get_movie_title(movie_id))

    return recommendations


# =========================
# API ENDPOINT
# =========================

@app.get("/recommend")
def recommend(user_id: int, movie: str):
    if user_id not in user_movie_matrix.index:
        raise HTTPException(status_code=404, detail="User not found")

    results = hybrid_recommend(user_id, movie)

    return {
        "user_id": user_id,
        "movie": movie,
        "recommendations": results
    }


# =========================
# ROOT CHECK
# =========================

@app.get("/")
def home():
    return {"message": "Movie Recommendation API is running"}