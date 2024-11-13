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

org_len = len(ratings)

# Remove all ratings that are less than 4
ratings_filtered = ratings.query("rating >= 4")

rating_filtered_len = len(ratings_filtered)

print(f"Ratings left after filtering: {rating_filtered_len}/{org_len}")

# Only take the top 1000 movies ordered by the number of ratings
top_movies = ratings_filtered["movieId"].value_counts().head(1000).index

ratings_filtered = ratings_filtered[ratings_filtered["movieId"].isin(top_movies)]

print(f"Movies left after filtering: {len(top_movies)}/{len(movie_descs)}")


# %%

# Create an edgelist from the dataframe
edges = defaultdict(lambda: 0)

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
for local_edges in tqdm(results, desc="Combining results"):
    for key, value in local_edges.items():
        edges[key] += value

# %%
# Convert the edges to a pyarrow table
import pyarrow as pa

print("Save the edges to a parquet file")

edges_table = pa.Table.from_pandas(
    pd.DataFrame(
        [
            {"movie1": key[0], "movie2": key[1], "weight": value}
            for key, value in edges.items()
        ]
    )
)

# %%
# Save the table to a file
edges_table.to_pandas().to_parquet("data/edges.parquet")
