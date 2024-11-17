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
ratings_filtered = ratings.query("rating > 4")

rating_filtered_len = len(ratings_filtered)

print(f"Ratings left after filtering: {rating_filtered_len}/{org_len}")

# Only take the top 1000 movies ordered by the number of ratings
top_movies = ratings_filtered["movieId"].value_counts().head(1).index

ratings_filtered = ratings_filtered[ratings_filtered["movieId"].isin(top_movies)]

print(f"Movies left after filtering: {len(top_movies)}/{len(movie_descs)}")


# %%

# Create an edgelist from the dataframe
edges = defaultdict(lambda: 0)

users_per_movie = ratings_filtered.groupby("movieId").apply(
    lambda x: x["userId"].values
)

# %%
from itertools import combinations
from joblib import Parallel, delayed


def process_users(users):
    local_edges = defaultdict(lambda: 0)
    print(len(users))
    # combs = list(combinations(users, 2))

    # print("Number of combinations: ", len(combs))

    # for u1, u2 in tqdm(combs, desc="Local combinations", leave=False):
    #    key = tuple(sorted((u1, u2)))
    #    local_edges[key] += 1

    # print(f"Number of edges: {len(local_edges)}")
    return local_edges


results = [
    process_users(users) for users in tqdm(users_per_movie, desc="Creating edges")
]

# # Combine the results
# for local_edges in tqdm(results, desc="Combining results"):
#     for key, value in local_edges.items():
#         edges[key] += value

# print("Done")

# # %%

# print(f"Number of edges: {len(edges)}")

# # %%
# # Convert the edges to a pyarrow table
# import pyarrow as pa

# print("Save the edges to a parquet file")

# edges_table = pa.Table.from_pandas(
#     pd.DataFrame(
#         [{"u1": key[0], "u2": key[1], "weight": value} for key, value in edges.items()]
#     )
# )

# # %%
# # Save the table to a file
# edges_table.to_pandas().to_parquet("data/edges_users.parquet")
