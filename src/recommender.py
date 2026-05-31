import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    return matrix.fillna(0.0)


def compute_similarity(matrix: pd.DataFrame) -> pd.DataFrame:
    similarity_array = cosine_similarity(matrix)
    similarity_df = pd.DataFrame(similarity_array, index=matrix.index, columns=matrix.index)
    return similarity_df


def recommend_for_user(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    similarity: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    user_ratings = ratings[ratings["userId"] == user_id]
    watched_movies = set(user_ratings["movieId"])

    similar_scores = similarity.loc[user_id].drop(index=user_id)
    similar_users = similar_scores[similar_scores > 0].sort_values(ascending=False)

    score_table = {}
    for similar_user_id, similarity_score in similar_users.items():
        similar_user_ratings = ratings[ratings["userId"] == similar_user_id]
        for _, row in similar_user_ratings.iterrows():
            movie_id = int(row["movieId"])
            if movie_id in watched_movies:
                continue
            score_table.setdefault(movie_id, 0.0)
            score_table[movie_id] += similarity_score * row["rating"]

    if not score_table:
        return pd.DataFrame(columns=["title", "genres", "score"])

    scored_movies = pd.DataFrame(
        [(movie_id, score) for movie_id, score in score_table.items()],
        columns=["movieId", "score"],
    )
    scored_movies = scored_movies.sort_values(by="score", ascending=False).head(top_n)
    return scored_movies.merge(movies, on="movieId")[ ["title", "genres", "score"] ]


def get_popular_movies(ratings: pd.DataFrame, movies: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    popularity = (
        ratings.groupby("movieId")["rating"].mean()
        .reset_index()
        .merge(ratings.groupby("movieId")["rating"].count().reset_index(), on="movieId", suffixes=("_mean", "_count"))
    )
    popularity = popularity[popularity["rating_count"] >= 20]
    popularity = popularity.sort_values(["rating_mean", "rating_count"], ascending=[False, False]).head(top_n)
    return popularity.merge(movies, on="movieId")[ ["title", "genres", "rating_mean", "rating_count"] ]
