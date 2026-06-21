import pandas as pd
import numpy as np

df = pd.read_csv('C:/FastAPIs for ml/Research_ollama/arxiv_data.csv')

df.nunique()

df = df.drop_duplicates()

df = df.drop(columns = ['terms'])

import re

corpus = []

for i in range(0,20000):
  paper = re.sub('[^a-zA-Z]',' ', df.iloc[i,-1])
  paper = paper.lower()
  paper = paper.split()

  paper = ' '.join(paper)
  corpus.append(paper)

from sentence_transformers import SentenceTransformer
papers = SentenceTransformer('all-MiniLM-L6-v2')
x = papers.encode(corpus, batch_size = 1000, show_progress_bar = True)

x.shape

query = "astro physics"
query_embed = papers.encode(query)
query_embed.shape

from sklearn.metrics.pairwise import cosine_similarity

cs = cosine_similarity(query_embed.reshape(1,-1),x)

tops = cs[0].argsort()[::-1]

print(df.iloc[tops[:10]])

x = np.array(x).astype('float32')

# Replacement to Cosine_similarity
import faiss
faiss_index = faiss.IndexFlatL2(384)
faiss_index.add(x)

query = "NLP transformers"
query_embed = papers.encode([query])
query_embed.shape

query_embed = query_embed.astype('float32')

Distances, indexes = faiss_index.search(query_embed,10) # Top-10 retrievals on similarity

import joblib as jb

faiss.write_index(faiss_index,"Faiss_indexes")
df = df.iloc[:20000]
jb.dump(df,"Papers")