## run kmeans using given labels

import numpy as np
embeddings = np.load("/work3/s204161/embeddings.npy")
embeddings = embeddings.reshape(embeddings.shape[0],embeddings.shape[1]*embeddings.shape[2])
print(embeddings.shape)
from sklearn.cluster import MiniBatchKMeans

# dont standardize embeddings vector to have mean 0 and std 1 ... because for our hidden weights, some features ARE more important than others...
# embeddings = embeddings - embeddings.mean(axis=0)
# embeddings = embeddings / embeddings.std(axis=0)

# Define MiniBatch K-Means
mini_batch_kmeans = MiniBatchKMeans(n_clusters=200, batch_size=1000, random_state=42)
mini_batch_kmeans.fit(embeddings)

# Get cluster labels
import copy
labels = copy.deepcopy(mini_batch_kmeans.labels_)
centers = copy.deepcopy(mini_batch_kmeans.cluster_centers_)


## for sparse clusters of length < 5 join them with neighbouring clusters
# get the number of movies in each cluster
unique, counts = np.unique(labels, return_counts=True)
cluster_counts = dict(zip(unique, counts))

# get the number of clusters
n_clusters = len(cluster_counts)
from tqdm import tqdm

# merge clusters with less than 5 movies
for cluster, count in tqdm(list(cluster_counts.items())):  # Iterate over a copy of the dictionary
    if count < 5:
        while True:  # Find a valid cluster to merge into
            distances = np.linalg.norm(centers - centers[cluster], axis=1)
            distances[cluster] = np.inf  # Avoid merging a cluster with itself
            closest_cluster = np.argmin(distances)
            
            # Check if the closest cluster is valid
            if closest_cluster in cluster_counts:
                break  # Exit the loop once a valid cluster is found
            
            # If not valid, continue searching
            distances[closest_cluster] = np.inf

        # merge the clusters
        labels[labels == cluster] = closest_cluster
        
        # update the cluster counts
        cluster_counts[closest_cluster] += count
        cluster_counts.pop(cluster)
        n_clusters -= 1
np.save("data/bert_clusters.npy",labels)
print(unique, counts = np.unique(labels, return_counts=True))