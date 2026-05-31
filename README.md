# AI-Powered Movie Recommendation System

A polished Streamlit application that uses the MovieLens dataset to deliver personalized movie recommendations with a professional landing page and analytics dashboard.

## Features

- AI-style collaborative filtering recommendations for sample MovieLens users
- Modern landing page and dashboard layout
- Search movies by title with average rating and review count
- Personalized user profile panel with rated movie summary
- Popular and top-rated movie insights
- Footer branding by `M Deepan vel`

## Project structure

- `app.py` - Streamlit dashboard entry point and UI layout
- `src/data_loader.py` - MovieLens download and dataset loader
- `src/recommender.py` - recommendation engine and scoring logic
- `requirements.txt` - Python dependencies
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

- Select a user from the sidebar to generate personalized suggestions.
- Use the recommendation count slider to control how many movies appear.
- Search movie titles to explore ratings and genres.
- Review popular and high-rated movie sections for dataset insights.

## Notes

- The dataset downloads automatically into `data/ml-latest-small/`.
- Delete `data/ml-latest-small/` to force a fresh download.
- The app is designed to run locally with `streamlit run app.py`.
