import networkx as nx
import pandas as pd
import pickle
import numpy as np
from collections import defaultdict, Counter
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

with open("data/movie_graph.pickle", "rb") as f:
    G = pickle.load(f)


def predict_movies(G: nx.Graph, user_watched_movies: list[int], weighted: bool = True):
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

    # Iterate over each watched movie and process its neighbors
    for movie in user_watched_movies:
        for neighbor, edge_attrs in G.adj[movie].items():
            if neighbor not in user_watched_movies:
                weight = edge_attrs.get("weight", 1)
                recommendations[neighbor] += weight if weighted else 1

    # Convert recommendations to a DataFrame and sort by the chosen metric
    metric = "weight" if weighted else "count"
    predictions = (
        pd.DataFrame.from_dict(recommendations, orient="index")
        .reset_index()
        .rename(columns={"index": "movieId", 0: metric})
        .sort_values(by=metric, ascending=False)
    )

    return predictions["movieId"].tolist()


ratings = pd.read_csv("data/ml-32m/ratings.csv")
movie_descs = pd.read_csv("data/movies_with_description.csv")
ratings = ratings[ratings["movieId"].isin(movie_descs["movieId"])]
ratings = ratings[ratings["rating"] >= 5.0]

users_to_analyze = [304, 6741, 147001]

preds = {u: [] for u in users_to_analyze}
preds_weighted = {u: [] for u in users_to_analyze}

for user in users_to_analyze:
    movies_watched = ratings[ratings["userId"] == user]["movieId"].tolist()

    preds[user] = predict_movies(G, movies_watched, weighted=True)
    preds_weighted[user] = predict_movies(G, movies_watched, weighted=True)

with open("data/predictions.pickle", "wb") as f:
    pickle.dump(preds, f)

with open("data/predictions_weighted.pickle", "wb") as f:
    pickle.dump(preds_weighted, f)

# sample 1000 users that have at least 5 ratings
users_to_analyze = (
    ratings["userId"]
    .value_counts()[ratings["userId"].value_counts() >= 5]
    .sample(1000)
    .index
)

accuracies = []
accuracies_weighted = []

K = 8
TEST_SIZE = 0.2

for user in tqdm(users_to_analyze, desc="Users"):
    movies_watched = ratings[ratings["userId"] == user]["movieId"].tolist()

    if len(movies_watched) < 5:
        continue

    train_movies, test_movies = train_test_split(
        movies_watched, test_size=TEST_SIZE, random_state=42
    )

    predicted_movies = predict_movies(G, train_movies, weighted=False)[:K]
    predicted_movies_weighted = predict_movies(G, train_movies, weighted=True)[:K]

    correct_predictions = any(movie in test_movies for movie in predicted_movies)
    correct_predictions_weighted = any(
        movie in test_movies for movie in predicted_movies_weighted
    )

    accuracies.append(correct_predictions)
    accuracies_weighted.append(correct_predictions_weighted)

accuracy = sum(accuracies) / len(accuracies)
print(f"Accuracy: {accuracy:.2f}")

accuracy_weighted = sum(accuracies_weighted) / len(accuracies_weighted)
print(f"Accuracy (weighted): {accuracy_weighted:.2f}")
