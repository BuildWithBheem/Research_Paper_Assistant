from pydantic import BaseModel
from fastapi import FastAPI
import re
import ollama
from sentence_transformers import SentenceTransformer
import joblib as jb
import faiss
from Evaluation_metrics import Embed
df = jb.load("Papers")
faiss_indexes = faiss.read_index("Faiss_indexes")
embed = SentenceTransformer('all-MiniLM-L6-v2')

class Research_Query(BaseModel):
    Query : str

app = FastAPI()

@app.post("/research-paper-assistant")

def Query(User_query : Research_Query):
    index = Embed(User_query.Query)

    paper1 = df.iloc[index[0][0]]
    paper2 = df.iloc[index[1][0]]
    paper3 = df.iloc[index[2][0]]
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
    options={"temperature" : 0.1}
    )
    result = ai_summary["message"]["content"]
    cleaned = re.sub(r'<think>.*?</think>',' ',result).strip()

    return {"Summary": cleaned}

@app.post("/Search")
def search(user_search: Research_Query):
    
    index = Embed(user_search.Query)

    top_papers = []

    for i in index:
        papers_index = i[0]
        row = df.iloc[papers_index]

        top_papers.append({
            "title": row["titles"],
            "abstracts":row["summaries"]
        })

    return {"Search": top_papers}