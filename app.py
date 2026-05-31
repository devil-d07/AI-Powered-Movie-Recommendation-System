import streamlit as st
import pandas as pd

from src.data_loader import download_movielens, prepare_data
from src.recommender import (
    build_user_item_matrix,
    compute_similarity,
    get_popular_movies,
    recommend_for_user,
)


st.set_page_config(page_title="MovieLens Recommender", layout="wide", page_icon="🎬")


def inject_styles():
    st.markdown(
        """
        <style>
        .hero-banner {padding: 2rem 2.5rem; border-radius: 1.2rem; background: linear-gradient(135deg, #0f172a, #1e3a8a); color: white;}
        .hero-banner h1 {font-size: 3rem; margin-bottom: 0.5rem;}
        .hero-banner p {font-size: 1.1rem; line-height: 1.8; margin-bottom: 1.25rem;}
        .hero-banner .badge {display:inline-block; padding:0.45rem 0.9rem; border-radius:999px; background:#2563eb; color:#fff; font-weight:600; margin-top:0.5rem;}
        .overview-card {padding:1.25rem; border-radius:1rem; background:#f8fafc; color:#0f172a; box-shadow:0 12px 30px rgba(15,23,42,0.08);}
        .overview-card h4, .overview-card p {color:#0f172a;}
        .overview-card strong {color:#0f172a;}
        .overview-grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:1rem;}
        .footer {padding:1.5rem 0; text-align:center; color:#64748b;}
        .footer a {color:#2563eb; text-decoration:none;}
        .section-title {font-size:1.3rem; font-weight:700; margin-bottom:0.5rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data():
    data_dir = download_movielens()
    ratings, movies = prepare_data(data_dir)
    return ratings, movies


@st.cache_data(show_spinner=False)
def build_models(ratings):
    matrix = build_user_item_matrix(ratings)
    similarity = compute_similarity(matrix)
    return matrix, similarity


def get_top_rated_movies(ratings: pd.DataFrame, movies: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    movie_stats = (
        ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    )
    movie_stats = movie_stats[movie_stats["count"] >= 30].sort_values(by=["mean", "count"], ascending=[False, False]).head(top_n)
    return movie_stats.merge(movies, on="movieId")[ ["title", "genres", "mean", "count"] ]


def get_user_profile(user_id: int, ratings: pd.DataFrame, movies: pd.DataFrame) -> dict:
    user_ratings = ratings[ratings["userId"] == user_id]
    rated_count = len(user_ratings)
    average_rating = round(user_ratings["rating"].mean(), 2) if rated_count else 0.0
    favorite_movies = (
        user_ratings.merge(movies, on="movieId")
        .sort_values(by=["rating"], ascending=False)
        .head(5)[["title", "genres", "rating"]]
    )
    return {
        "rated_count": rated_count,
        "average_rating": average_rating,
        "favorite_movies": favorite_movies,
    }


def render_header():
    st.markdown(
        """
        <div class='hero-banner'>
            <span class='badge'>AI-powered movie recommender</span>
            <h1>Discover movies based on real user taste</h1>
            <p>Explore the MovieLens dataset with an interactive recommendation engine, in-depth user insights, searchable movie discovery, and a modern analytics dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(ratings: pd.DataFrame, movies: pd.DataFrame, user_profile: dict):
    st.write("")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total users", int(ratings["userId"].nunique()))
    col2.metric("Total movies", int(movies["movieId"].nunique()))
    col3.metric("Total ratings", int(len(ratings)))
    col4.metric("Your average rating", user_profile["average_rating"])


def render_recommendation_section(recommendations: pd.DataFrame):
    if recommendations.empty:
        st.warning("No recommendations available yet. Try a different user or refresh the dataset.")
    else:
        st.dataframe(recommendations.rename(columns={"score": "recommendation score"}).reset_index(drop=True), use_container_width=True)


def render_search_section(movies: pd.DataFrame, ratings: pd.DataFrame, query: str):
    if not query:
        return

    results = movies[movies["title"].str.contains(query, case=False, na=False)].copy()
    if results.empty:
        st.info(f"No movies found matching '{query}'.")
        return

    stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    results = results.merge(stats, on="movieId", how="left").fillna({"mean": 0.0, "count": 0})
    results = results.sort_values(by=["mean", "count"], ascending=[False, False]).head(12)
    st.markdown(f"### Search results for '{query}'")
    st.dataframe(
        results.rename(columns={"mean": "avg rating", "count": "rating count"})[["title", "genres", "avg rating", "rating count"]],
        use_container_width=True,
    )


def render_dashboard(ratings: pd.DataFrame, movies: pd.DataFrame, recommendations: pd.DataFrame, user_profile: dict, search_query: str):
    render_header()

    st.write("---")
    render_metrics(ratings, movies, user_profile)

    with st.container():
        st.write("")
        col1, col2 = st.columns((2, 1))
        with col1:
            st.subheader("Your personalized recommendations")
            render_recommendation_section(recommendations)

        with col2:
            st.subheader("Your current profile")
            st.markdown(
                f"<div class='overview-card'>"
                f"<h4 class='section-title'>User stats</h4>"
                f"<p><strong>Rated movies</strong>: {user_profile['rated_count']}</p>"
                f"<p><strong>Average rating</strong>: {user_profile['average_rating']}</p>"
                f"<h4 class='section-title'>Top rated by you</h4>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(user_profile["favorite_movies"], use_container_width=True)

    st.write("---")
    st.subheader("Popular movies from MovieLens")
    popular = get_popular_movies(ratings, movies, top_n=8)
    st.dataframe(popular.rename(columns={"rating_mean": "avg rating", "rating_count": "count"}), use_container_width=True)

    st.write("---")
    st.subheader("Top high-rated movies")
    top_rated = get_top_rated_movies(ratings, movies, top_n=8)
    st.dataframe(top_rated.rename(columns={"mean": "avg rating", "count": "rating count"}), use_container_width=True)

    if search_query:
        st.write("---")
        render_search_section(movies, ratings, search_query)

    st.write("---")
    st.markdown(
        "<div class='footer'>Built by <strong>M Deepan vel</strong> — MovieLens recommender dashboard with personalized suggestions and search.</div>",
        unsafe_allow_html=True,
    )


def main():
    inject_styles()
    ratings, movies = load_data()
    matrix, similarity = build_models(ratings)

    st.sidebar.title("Explore recommendations")
    st.sidebar.markdown("Choose a user, search titles, and tune how many suggestions appear in the dashboard.")

    user_options = sorted(ratings["userId"].unique())
    user_id = st.sidebar.selectbox("User ID", user_options, index=0)
    top_n = st.sidebar.slider("Recommendation count", min_value=5, max_value=20, value=10)
    search_query = st.sidebar.text_input("Search movie titles")

    recommendations = recommend_for_user(user_id, ratings, movies, similarity, top_n=top_n)
    user_profile = get_user_profile(user_id, ratings, movies)

    render_dashboard(ratings, movies, recommendations, user_profile, search_query)


if __name__ == "__main__":
    main()
