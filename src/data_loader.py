import os
import pathlib
import zipfile
from urllib.parse import urljoin

import pandas as pd
import requests

MOVIELENS_BASE = "https://files.grouplens.org/datasets/movielens/"
SMALL_ZIP = "ml-latest-small.zip"


def download_movielens(target_dir: str = "data", force: bool = False) -> pathlib.Path:
    target_dir_path = pathlib.Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    zip_path = target_dir_path / SMALL_ZIP
    extracted_dir = target_dir_path / "ml-latest-small"

    if force or not extracted_dir.exists():
        response = requests.get(urljoin(MOVIELENS_BASE, SMALL_ZIP), stream=True)
        response.raise_for_status()

        with open(zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(path=target_dir_path)

    return extracted_dir


def load_ratings(data_dir: str = "data/ml-latest-small") -> pd.DataFrame:
    path = pathlib.Path(data_dir) / "ratings.csv"
    return pd.read_csv(path)


def load_movies(data_dir: str = "data/ml-latest-small") -> pd.DataFrame:
    path = pathlib.Path(data_dir) / "movies.csv"
    return pd.read_csv(path)


def prepare_data(data_dir: str = "data/ml-latest-small") -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)
    return ratings, movies
