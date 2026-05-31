# AI-Powered Movie Recommendation System

A simple end-to-end Python project that builds a collaborative filtering recommender using the MovieLens dataset and exposes an interactive Streamlit dashboard.

## Features

- Collaborative filtering based on MovieLens ratings
- Personalized movie recommendations for sample users
- Streamlit dashboard with dataset metrics and popular movie insights
- Automatic dataset download and preprocessing

## Project structure

- `app.py` - Streamlit dashboard entry point
- `src/data_loader.py` - dataset download and loading utilities
- `src/recommender.py` - recommendation engine and ranking logic
- `requirements.txt` - project dependencies
- `.gitignore` - ignored files for local development

## Setup

1. Create a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
streamlit run app.py
```

## Usage

- Use the sidebar to select a `User ID` from the MovieLens dataset.
- Adjust the number of recommendations.
- View popular movies and personalized suggestions in the dashboard.

## Notes

- The project downloads the MovieLens `ml-latest-small` dataset automatically into `data/`.
- If you want to refresh the dataset, delete `data/ml-latest-small/` and rerun the app.
