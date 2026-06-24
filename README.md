# Research Paper Assistant

An AI-powered Research Paper Assistant that helps users discover, retrieve, and understand academic papers using semantic search, vector databases, and Large Language Models (LLMs).

## Overview

Research papers contain valuable information, but finding relevant papers and extracting useful insights can be time-consuming. This project solves that problem by combining modern Natural Language Processing (NLP) techniques with Retrieval-Augmented Generation (RAG).

The system retrieves the most relevant research papers based on the user's query, ranks them using semantic similarity, and generates concise AI-powered explanations and summaries.

## Features

* Semantic paper search using sentence embeddings
* Fast retrieval using FAISS vector indexing
* Context-aware AI responses using an LLM
* Research paper recommendations based on user queries
* Interactive Streamlit web interface
* FastAPI backend for scalable deployment
* Retrieval-Augmented Generation (RAG) pipeline
* Local LLM integration through Ollama
* Re-ranking using Cross-Encoder for better retrieval

## Architecture

```text
User Query
     │
     ▼
Sentence Transformer
(all-MiniLM-L6-v2)
     │
     ▼
Query Embedding
     │
     ▼
FAISS Vector Search
     │
     ▼
Cross-Encoder Re-ranking
     │
     ▼
Top-K relevant papers
     │
     ▼
Context Construction
     │
     ▼
Qwen LLM (Ollama)
     │
     ▼
Generated Response
```

## Tech Stack

### Backend

* Python
* FastAPI
* Pydantic

### Machine Learning & NLP

* Sentence Transformers
* FAISS
* Scikit-Learn
* NumPy
* Pandas
* CrossEncoder

### LLM

* Ollama
* Qwen3-4B

### Frontend

* Streamlit

## Dataset

The system uses research paper abstracts from the arXiv dataset.

Each paper contains:

* Title
* Abstract (Summaries)
* Metadata

The abstracts are converted into dense vector embeddings using Sentence Transformers and stored inside a FAISS vector index for efficient similarity search.

## How It Works

### Step 1: Embedding Generation

Research paper abstracts are converted into dense vector representations using:

```python
all-MiniLM-L6-v2
```

This enables semantic understanding rather than simple keyword matching.

### Step 2: Vector Indexing

The generated embeddings are stored inside a FAISS index:

```python
faiss.IndexFlatL2
```

This allows efficient retrieval from thousands of research papers.

### Step 3: Retrieval

When a user enters a query:

1. The query is embedded.
2. Similar papers are retrieved from FAISS.
3. Retrieved papers are re-ranked.
3. Top relevant papers are selected.

### Step 4: AI-Assisted Explanation

The retrieved papers are provided as context to the LLM.

The LLM:

* Summarizes findings
* Explains concepts
* Answers user questions
* Generates concise responses grounded in retrieved papers

## Example Query

```text
How are vector databases used in Retrieval-Augmented Generation?
```

### Retrieved Papers

* Efficient Similarity Search in High Dimensions
* Dense Passage Retrieval
* Retrieval-Augmented Generation for Knowledge Intensive NLP Tasks

### Generated Output

The assistant explains how vector embeddings are stored in a vector database, retrieved through similarity search, and supplied to an LLM to improve factual accuracy and reduce hallucinations.

## Future Improvements

* Cross-Encoder Re-ranking ✅
* Hybrid Search (BM25 + Embeddings)
* PDF Upload Support
* Citation Generation
* Multi-document Summarization
* Cloud Deployment
* Conversational Memory

## Installation

```bash
git clone https://github.com/your-username/research-paper-assistant.git

cd research-paper-assistant

pip install -r requirements.txt
```

Run FastAPI:

```bash
uvicorn main:app --reload
```

Run Streamlit:

```bash
streamlit run app.py
```

## Key Learnings

* Semantic Search
* Vector Databases
* Retrieval-Augmented Generation (RAG)
* FastAPI Development
* Streamlit Application Development
* LLM Integration
* Embedding-Based Information Retrieval

## Project Impact

This project demonstrates the practical implementation of modern AI systems by combining information retrieval, vector search, and Large Language Models into a complete end-to-end application. It highlights skills in Machine Learning, NLP, Backend Development, API Design, and AI System Engineering.

---

Built by **Bhimaraju Sai Koundinya** as part of continuous exploration in Machine Learning, NLP, and AI Systems.