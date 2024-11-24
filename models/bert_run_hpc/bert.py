## Memory issues if not using HPC, if trying to run on all 50k movies.


#!pip install kagglehub
import kagglehub
import os
import shutil
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import pickle

#from datasketch import MinHash, MinHashLSH
def load_data():
    # Contains movieId, title, genres   
    ml_movies = pd.read_csv('data/ml-32m/movies.csv')
    
    # Contains userId, movieId, rating, timestamp
    ml_ratings = pd.read_csv('data/ml-32m/ratings.csv')
    
    # Contains userId, movieId, tag, timestamp
    ml_tags = pd.read_csv('data/ml-32m/tags.csv')

    # Contains movieId, imdbId, tmdbId
    ml_links = pd.read_csv('data/ml-32m/links.csv')

    # IMDB Dataset:
    imdb = pd.read_csv('data/imdb/movies.csv')

    return ml_movies, ml_ratings, ml_tags, ml_links, imdb

ml_movies, ml_ratings, ml_tags, ml_links, imdb = load_data()


# Preprocessing IMDB:

#drop if description is "Add a plot" because we have no description then
imdb_descriptions = imdb[imdb['description'] != "Add a Plot"]

imdb_descriptions['id'] = imdb_descriptions['id'].str[2:]
imdb_descriptions['id'] = pd.to_numeric(imdb_descriptions['id'], errors='coerce')

#keep what we need
imdb_descriptions = imdb_descriptions[['id', 'description']]

#merge them, so we have a li nk between the two datasets
merged_links = ml_links.merge(imdb_descriptions, left_on='imdbId', right_on='id', how='inner').dropna(subset=['id'])

#merge the movies with the descriptions if that is what we want
ml_movies_description = ml_movies.merge(merged_links, left_on='movieId', right_on='movieId', how='inner')

#ml_movies_description.to_csv('data/movies_with_description.csv', index=False)
#load in bert encoding model from huggingface

from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
inputs = tokenizer(ml_movies_description['description'].tolist(), return_tensors="pt", padding=True, truncation=True)

import matplotlib.pyplot as plt
#histogram of the length of the description in characters
ml_movies_description['description'].apply(lambda x: len(x)).hist()

#histogram of the length of the description in words
ml_movies_description['description'].apply(lambda x: len(x.split())).hist()


#histogram of the length of the description in tokens
ml_movies_description['description'].apply(lambda x: len(tokenizer(x)['input_ids'])).hist()

#add legend for each histogram
plt.legend(['Characters','Words', 'Tokens'])
plt.xlabel('Token count')
plt.ylabel('Number of descriptions')
plt.title('Histogram of the length of the description in tokens')
plt.savefig('figs/histogram_tokenize_bert.png')


#run inpquts through the model in batches to avoid memory issues and also pass attention masks
import torch
from tqdm import tqdm
outputs = []
model = model.to("cuda")
inputs = inputs.to("cuda")
batch_size = 64
#without saving gradients to avoid gpu memory issues/leaks
# if outputs file exists, load it, else run the model
if os.path.exists('/work3/s204161/embeddings.npy'):
    outputs = torch.load('/work3/s204161/embeddings.npy')
else:
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs['input_ids']), batch_size)):
            batch = inputs['input_ids'][i:i+batch_size]
            attention_mask = inputs['attention_mask'][i:i+batch_size]
            output = model(batch, attention_mask=attention_mask).last_hidden_state
            #output = model(batch, attention_mask=attention_mask)['logits']

            #send the output to the cpu afterwards to avoid memory issues on the gpu
            outputs.append(output.detach().cpu())

    #make list of tensors into one tensor
    outputs = torch.cat(outputs)
    np.save('/work3/s204161/embeddings.npy', outputs.numpy())

print("done...")
