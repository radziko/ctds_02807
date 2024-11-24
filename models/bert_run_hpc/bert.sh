#first run bert.py to generate the embeddings, then kmeans.py to cluster the embeddings, then bert_partition to partition embeddings array using clusters

source /work3/s204161/my_env/some_env/bin/activate

python models/bert_run_hpc/bert.py