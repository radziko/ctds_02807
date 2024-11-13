# %%
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
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


def process_group(movie_id, group):
    local_edges = defaultdict(lambda: 0)
    users = group["userId"].values
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            local_edges[(users[i].item(), users[j].item())] += 1
            local_edges[(users[j].item(), users[i].item())] += 1
    return local_edges


cpus = os.cpu_count()

with ThreadPoolExecutor(max_workers=cpus) as executor:
    print("Creating tasks")
    futures = {
        executor.submit(process_group, movie_id, group): movie_id
        for movie_id, group in tqdm(
            ratings_filtered.groupby("movieId"), desc="Creating tasks"
        )
    }
    print("Processing ratings")
    for future in tqdm(
        as_completed(futures), total=len(futures), desc="Processing ratings"
    ):
        local_edges = future.result()
        for key, value in local_edges.items():
            edges[key] += value


# Save the edges to a pickle file
with open("data/edges.pkl", "wb") as f:
    pickle.dump(edges, f)
