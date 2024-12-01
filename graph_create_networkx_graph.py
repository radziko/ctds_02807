# %%
import pandas as pd
import networkx as nx
from tqdm import tqdm
import pickle

edgelist = pd.read_parquet("data/edges-movie-all.parquet")
# %%

# Create a graph from the edgelist
G = nx.Graph()

total = len(edgelist)

for _, row in tqdm(edgelist.iterrows(), total=total):
    G.add_edge(row["movie1"], row["movie2"], weight=row["weight"])

# %%
# Save the graph to a pickle
with open("data/movie_graph.pickle", "wb") as f:
    pickle.dump(G, f)
