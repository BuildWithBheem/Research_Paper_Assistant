import joblib
import ranx
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer,CrossEncoder
df = joblib.load('Papers')
index = faiss.read_index('Faiss_indexes')

embed = SentenceTransformer('all-MiniLM-L6-v2')
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def Embed(query):
    emb_query = embed.encode([query])
    emb_query = np.array(emb_query).astype('float32')

    dist, indexes = index.search(emb_query, 10)

    retr_papers = []
    for i in indexes[0]:
        retr_papers.append(df.iloc[i]['summaries'])

    pairs = [(query,i) for i in retr_papers]

    score = ce.predict(pairs)

    ranking = []
    for i in range(len(retr_papers)):
        ranking.append([indexes[0][i],score[i]])

    ranking.sort(key= lambda x:x[1], reverse = True)

    return ranking