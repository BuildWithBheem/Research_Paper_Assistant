from pydantic import BaseModel
from fastapi import FastAPI
import re
import ollama
from sentence_transformers import SentenceTransformer
import joblib as jb
import faiss

df = jb.load("Papers")
faiss_indexes = faiss.read_index("Faiss_indexes")
embed = SentenceTransformer('all-MiniLM-L6-v2')

class Research_Query(BaseModel):
    Query : str

app = FastAPI()

@app.post("/research-paper-assistant")

def Query(User_query : Research_Query):
    query_embed = embed.encode([User_query.Query])
    query_embed = query_embed.astype('float32')

    dist, index = faiss_indexes.search(query_embed,3)

    paper1 = df.iloc[index[0,0]]
    paper2 = df.iloc[index[0,1]]
    paper3 = df.iloc[index[0,2]]
    prompt = f"""answer the user {User_query.Query} within 50 words, with the context of {paper1},{paper2},{paper3}
                You may only answer using the retrieved papers.
                If the retrieved papers do not contain enough information,
                state that the information is unavailable in the dataset.
                
                Do not use outside knowledge."""
    
    ai_summary = ollama.chat(
    model = 'qwen2.5:1.5b',
    messages= [
        {"role":"system", "content":"You are a research paper assistant, dont use Markdown format...just plain text"},
        {"role":"user","content":prompt}
    ],
    options={"think" : False}
    )
    result = ai_summary["message"]["content"]
    cleaned = re.sub(r'<think>.*?</think>',' ',result).strip()

    return {"Summary": cleaned}

@app.post("/Search")
def search(user_search: Research_Query):
    query_embed = embed.encode([user_search.Query])
    query_embed = query_embed.astype('float32')

    dist, index = faiss_indexes.search(query_embed,10)

    top_papers = []

    for i in index[0]:
        row = df.iloc[i]

        top_papers.append({
            "title": row["titles"],
            "abstracts":row["summaries"]
        })

    return {"Search": top_papers}