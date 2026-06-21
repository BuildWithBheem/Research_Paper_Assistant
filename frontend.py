import streamlit as st
import requests

FASTAPI_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.paper-box {
    background-color: #1e1e2e;
    padding: 14px 18px;
    border-radius: 8px;
    margin-bottom: 10px;
    border: 1px solid #313244;
}
.paper-title {
    color: #cdd6f4 !important;
    font-size: 15px;
    font-weight: 600;
    margin: 0;
}
.rank-badge {
    color: #a78bfa;
    font-size: 13px;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

st.title("Research Assistant")
st.caption("Semantic search over 20,000 ArXiv papers, powered by FAISS + Ollama")

if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Search papers:", placeholder="e.g., NLP transformers, monocular depth estimation")

if st.button("Search", type="primary"):
    if user_input.strip():
        with st.spinner("Searching..."):
            try:
                res = requests.post(f"{FASTAPI_URL}/Search", json={"Query": user_input})
                if res.status_code == 200:
                    st.session_state.search_results = res.json().get("Search", [])
                else:
                    st.error(f"Search failed: {res.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Is uvicorn running?")
    else:
        st.warning("Please enter a search query.")

if st.session_state.search_results:
    st.markdown(f"**{len(st.session_state.search_results)} papers found**")
    for i, paper in enumerate(st.session_state.search_results, 1):
        title = paper.get("title", f"Paper {i}")
        abstract = paper.get("abstracts", "No abstract available.")

        with st.expander(f"#{i} :  {title}"):
            st.markdown(f"<p style='color:#9ca3af; font-size:13px; line-height:1.6'>{abstract}</p>", unsafe_allow_html=True)

with st.sidebar:
    st.sidebar.title("🤖 Chat with AI")
    st.caption("Uses top 3 retrieved papers as context")
    st.divider()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if chat_query := st.chat_input("Ask about the papers..."):
        st.session_state.chat_history.append({"role": "user", "content": chat_query})
        with st.chat_message("user"):
            st.markdown(chat_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    res = requests.post(
                        f"{FASTAPI_URL}/research-paper-assistant",
                        json={"Query": chat_query}
                    )
                    if res.status_code == 200:
                        answer = res.json().get("Summary", "No response.")
                        st.markdown(answer)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer
                        })
                    else:
                        st.error(f"Error: {res.status_code}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")