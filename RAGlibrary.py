import streamlit as st
import os
import json
import datetime
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


LIBRARY_FOLDER = "library"
INDEX_FILE = "library_index.json"
TOP_K = 3


def load_metadata():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r") as f:
            return json.load(f)
    return {"documents": [], "chunks": []}

def save_metadata(meta):
    with open(INDEX_FILE, "w") as f:
        json.dump(meta, f, indent=2)

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def process_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

if "meta" not in st.session_state:
    st.session_state.meta = load_metadata()

meta = st.session_state.meta


st.title("Personal RAG Library")

# Sidebar
with st.sidebar:
    st.header("Library")
    docs = meta["documents"]
    if docs:
        for d in docs:
            st.write(f"**{d['title']}** — {d['uploaded_at']}")
    else:
        st.write("empty")

    st.markdown("---")
    top_k = st.slider("Top-K retrieval", 1, 10, TOP_K)
    use_openai = st.checkbox("Use OpenAI (if key set)", value=False)

# Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    text = process_pdf(uploaded_file)
    chunks = chunk_text(text)

    
    meta["documents"].append({
        "title": uploaded_file.name,
        "uploaded_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    for c in chunks:
        meta["chunks"].append({"text": c, "source": uploaded_file.name})

   
    st.session_state.meta = meta
    save_metadata(meta)
    st.success(f"Added {uploaded_file.name} to your library!")


query = st.text_input("Ask a question about your library")
if query:
    texts = [c["text"] for c in meta["chunks"]]
    if texts:
        vectorizer = TfidfVectorizer().fit(texts + [query])
        vecs = vectorizer.transform(texts)
        qvec = vectorizer.transform([query])
        sims = cosine_similarity(qvec, vecs).flatten()
        top_idx = sims.argsort()[-top_k:][::-1]

        st.subheader("Retrieved Chunks")
        for i in top_idx:
            st.write(f"**Source:** {meta['chunks'][i]['source']}")
            st.write(meta["chunks"][i]["text"][:500] + "...")
            st.markdown("---")
    else:
        st.warning("No documents in library yet.")
