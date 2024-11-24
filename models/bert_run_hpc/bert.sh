#first run bert.py to generate the embeddings, then kmeans.py to cluster the embeddings, then bert_partition to partition embeddings array using clusters
source /dtu/blackhole/07/155527/camproj/env_works/bin/activate

python models/bert_run_hpc/bert.py