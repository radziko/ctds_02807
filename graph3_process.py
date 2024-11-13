import cugraph
import cudf

# Read the data
# We have a parquet table with the edges with columns "movie1", "movie2", "weight"

edges = cudf.read_parquet("data/edges.parquet")

# Create graph
G = cugraph.Graph()
G.from_cudf_edgelist(edges, source="movie1", destination="movie2", edge_attr="weight")

# Calculate weighted jaccard similarity

jaccard_weighted = cugraph.jaccard(G, use_weights=True)

# Write the results
jaccard_weighted.to_pandas().to_parquet("data/jaccard_weighted.parquet")
