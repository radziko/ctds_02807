# %%
import pandas as pd
import networkx as nx
from tqdm import tqdm

edgelist = pd.read_parquet("data/edges.parquet")
# %%

# Create a graph from the edgelist
G = nx.Graph()

for _, row in tqdm(edgelist.iterrows()):
    G.add_edge(row["movie1"], row["movie2"], weight=row["weight"])
