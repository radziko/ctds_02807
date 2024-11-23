import networkx as nx
import pandas as pd
import pickle
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("Loading graph...", flush=True)
with open("data/movie_graph.pickle", "rb") as f:
    G = pickle.load(f)
print("Graph loaded!", flush=True)


def predict_movies(
    G: nx.Graph,
    user_watched_movies: list[int],
    weighted: bool = True,
    combination: bool = False,
):
    """
    Predicts movies for a user based on their watched movies and a graph of movie relationships.

    Args:
        G: Graph representing movie relationships. Each edge can have a weight.
        user_watched_movies: List of movies watched by the user.
        weighted: Boolean indicating whether predictions should consider edge weights.

    Returns:
        A sorted list of recommended movie IDs based on their predicted relevance.
    """
    # Use a single dictionary for neighbor counts or weights
    recommendations = defaultdict(float)
    recommendations_weighted = defaultdict(float)

    # Iterate over each watched movie and process its neighbors
    for movie in user_watched_movies:
        for neighbor, edge_attrs in G.adj[movie].items():
            if neighbor not in user_watched_movies:
                weight = edge_attrs.get("weight", 1)
                recommendations[neighbor] += 1
                recommendations_weighted[neighbor] += weight

    # Convert recommendations to a DataFrame and sort by the chosen metric
    predictions = (
        pd.DataFrame.from_dict(recommendations, orient="index")
        .reset_index()
        .rename(columns={"index": "movieId", 0: "count"})
        .sort_values(by="count", ascending=False)
    )

    predictions_weighted = (
        pd.DataFrame.from_dict(recommendations, orient="index")
        .reset_index()
        .rename(columns={"index": "movieId", 0: "weight"})
        .sort_values(by="weight", ascending=False)
    )

    if combination:
        new_predictions = predictions.merge(
            predictions_weighted, on="movieId", how="inner"
        )

        new_predictions["combined"] = (
            np.log(new_predictions["count"]) * new_predictions["weight"]
        )

        return new_predictions.sort_values(by="combined", ascending=False)[
            "movieId"
        ].tolist()

    if weighted:
        return predictions_weighted["movieId"].tolist()

    return predictions["movieId"].tolist()


print("Loading data...", flush=True)
ratings = pd.read_csv("data/ml-32m/ratings.csv")
movie_descs = pd.read_csv("data/movies_with_description.csv")
ratings = ratings[ratings["movieId"].isin(movie_descs["movieId"])]
ratings = ratings[ratings["rating"] >= 5.0]
print("Data loaded!", flush=True)


print("Analysing predetermined users...", flush=True)
users_to_analyze = [304, 6741, 147001]

preds = {u: [] for u in users_to_analyze}
preds_weighted = {u: [] for u in users_to_analyze}
preds_combined = {u: [] for u in users_to_analyze}

for user in users_to_analyze:
    movies_watched = ratings[ratings["userId"] == user]["movieId"].tolist()

    preds[user] = predict_movies(G, movies_watched, weighted=True)
    preds_weighted[user] = predict_movies(G, movies_watched, weighted=True)
    preds_combined[user] = predict_movies(G, movies_watched, combination=True)

with open("data/predictions.pickle", "wb") as f:
    pickle.dump(preds, f)

with open("data/predictions_weighted.pickle", "wb") as f:
    pickle.dump(preds_weighted, f)

with open("data/predictions_combined.pickle", "wb") as f:
    pickle.dump(preds_combined, f)

print("Predictions saved!", flush=True)

print("Analysing random users...", flush=True)

# sample 1000 users that have at least 5 ratings
users_to_analyze = (
    ratings["userId"]
    .value_counts()[ratings["userId"].value_counts() >= 5]
    .sample(1000)
    .index
)
user_ratings_count = {}

all_predicted_movies = []
all_predict_movies_weighted = []
all_predict_movies_combined = []
all_correct_predictions = []
all_correct_predictions_weighted = []
all_correct_predictions_combined = []
all_train_movies = []
all_test_movies = []

K_MAX = 50
TEST_SIZE = 2

for user in tqdm(users_to_analyze, desc="Users"):
    movies_watched = ratings[ratings["userId"] == user]["movieId"].tolist()

    if len(movies_watched) < 5:
        continue

    user_ratings_count[user] = len(movies_watched)

    train_movies, test_movies = train_test_split(
        movies_watched, test_size=TEST_SIZE, random_state=42
    )

    all_train_movies.append(train_movies)
    all_test_movies.append(test_movies)

    predicted_movies = predict_movies(G, train_movies, weighted=False)
    predicted_movies_weighted = predict_movies(G, train_movies, weighted=True)
    predicted_movies_combined = predict_movies(G, train_movies, combination=True)

    correct_predictions = any(
        movie in test_movies
        for movie in predicted_movies[: min(len(predicted_movies), K_MAX)]
    )
    correct_predictions_weighted = any(
        movie in test_movies
        for movie in predicted_movies_weighted[
            : min(len(predicted_movies_weighted), K_MAX)
        ]
    )
    correct_predictions_combined = any(
        movie in test_movies
        for movie in predicted_movies_combined[
            : min(len(predicted_movies_combined), K_MAX)
        ]
    )

    all_predicted_movies.append(predicted_movies[: min(len(predicted_movies), K_MAX)])
    all_predict_movies_weighted.append(
        predicted_movies_weighted[: min(len(predicted_movies_weighted), K_MAX)]
    )
    all_predict_movies_combined.append(
        predicted_movies_combined[: min(len(predicted_movies_combined), K_MAX)]
    )

    all_correct_predictions.append(correct_predictions)
    all_correct_predictions_weighted.append(correct_predictions_weighted)
    all_correct_predictions_combined.append(correct_predictions_combined)

with open("data/evaluation.pickle", "wb") as f:
    pickle.dump(
        {
            "users": users_to_analyze,
            "user_ratings_count": user_ratings_count,
            "train_movies": all_train_movies,
            "test_movies": all_test_movies,
            "correct_predictions": all_correct_predictions,
            "correct_predictions_weighted": all_correct_predictions_weighted,
            "correct_predictions_combined": all_correct_predictions_combined,
            "all_predicted_movies": all_predicted_movies,
            "all_predict_movies_weighted": all_predict_movies_weighted,
            "all_predict_movies_combined": all_predict_movies_combined,
        },
        f,
    )
