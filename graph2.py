# %%
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import pickle

# %%
ratings = pd.read_csv("data/ml-32m/ratings.csv")

# %%
movie_descs = pd.read_csv("data/movies_with_description.csv")

# Filter all ratings that not in movie_descs

ratings = ratings[ratings["movieId"].isin(movie_descs["movieId"])]

# %%

# Create an edgelist from the dataframe
edges = defaultdict(lambda: 0)

ratings_filtered = ratings.query("rating >= 4")
# %%

movies_per_user = ratings_filtered.groupby("userId").apply(
    lambda x: x["movieId"].values
)

# %%
from itertools import combinations
from joblib import Parallel, delayed


def process_movies(movies):
    local_edges = defaultdict(lambda: 0)
    for movie1, movie2 in combinations(movies, 2):
        key = tuple(sorted((movie1, movie2)))
        local_edges[key] += 1
    return local_edges


results = Parallel(n_jobs=-1)(
    delayed(process_movies)(movies)
    for movies in tqdm(movies_per_user, desc="Creating edges")
)

# Combine the results
for local_edges in results:
    for key, value in local_edges.items():
        edges[key] += value

# %%
# Save it to a pickle file

with open("data/edges.pkl", "wb") as f:
    pickle.dump(edges, f)
