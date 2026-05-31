import streamlit as st

from src.data_loader import download_movielens, prepare_data
from src.recommender import build_user_item_matrix, compute_similarity, get_popular_movies, recommend_for_user


st.set_page_config(page_title="Movie Recommender", layout="wide")


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


def render_dashboard(ratings, movies, recommendations):
    st.header("AI-Powered Movie Recommendation Dashboard")
    st.markdown(
        "This simple collaborative filtering app uses MovieLens data to deliver personalized movie suggestions. "
        "Choose an existing user or explore the most popular movies in the dataset."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total users", int(ratings["userId"].nunique()))
        st.metric("Total movies", int(ratings["movieId"].nunique()))
    with col2:
        st.metric("Total ratings", int(len(ratings)))
        st.metric("Dataset", "MovieLens 100K / small")

    st.subheader("Top popular movies")
    popular = get_popular_movies(ratings, movies, top_n=8)
    st.dataframe(popular.rename(columns={"rating_mean": "avg rating", "rating_count": "count"}), use_container_width=True)

    st.subheader("Personalized recommendations")
    if recommendations.empty:
        st.info("Select a valid user to see personalized movie recommendations.")
    else:
        st.dataframe(recommendations.rename(columns={"score": "recommendation score"}), use_container_width=True)


def main():
    ratings, movies = load_data()
    matrix, similarity = build_models(ratings)

    st.sidebar.title("Recommendation options")
    st.sidebar.markdown("Choose a movie-lover from the sample dataset to generate recommendations.")

    user_id = st.sidebar.number_input(
        "User ID", min_value=int(ratings["userId"].min()), max_value=int(ratings["userId"].max()), value=1, step=1
    )
    top_n = st.sidebar.slider("Number of recommendations", min_value=5, max_value=20, value=10)

    recommendations = recommend_for_user(user_id, ratings, movies, similarity, top_n=top_n)
    render_dashboard(ratings, movies, recommendations)


if __name__ == "__main__":
    main()
