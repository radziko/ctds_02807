embeddings_file = '/work3/s204161/embeddings.npy'
df_clusters = 'data/bert_clusters.csv'
output_directory = "/work3/s204161/data/clustered_embeddings"  # Output directory
import pandas as pd
import numpy as np
import os

def save_clustered_embeddings_to_folder(embeddings, cluster_file, output_dir):
    """
    Groups embeddings by cluster and saves them as .npy files in the specified directory.

    Parameters:
        embeddings (numpy.ndarray): Array of shape (N, 108, 768).
        cluster_file (str): Path to the .npy file containing cluster labels (array of shape (N,)).
        output_dir (str): Directory where the clustered .npy files will be saved.
    """
    # load df
    df_bert = pd.read_csv('./data/df_clusters.csv')
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check input consistency
    if len(embeddings) != len(df_bert):
        raise ValueError("The number of embeddings must match the number of cluster labels.")
    
    # Group embeddings by cluster
    cluster_names = np.unique(df_bert['cluster_label'])

    # Save each cluster's embeddings as a single .npy file
    for cluster_name in cluster_names:
        embedding_idx = df_bert[df_bert['cluster_label'] == cluster_name].index
        embeddings_npy = embeddings[embedding_idx]
        file_name = f"{cluster_name}.npy"  # Use the cluster number as the file name
        file_path = os.path.join(output_dir, file_name)
        np.save(file_path, embeddings_npy)
        print(f"Saved cluster {cluster_name} with shape {embeddings_npy.shape} to {file_path}")

# Load the embeddings array
embeddings = np.load(embeddings_file)

# Call the function
save_clustered_embeddings_to_folder(embeddings, df_clusters, output_directory)
