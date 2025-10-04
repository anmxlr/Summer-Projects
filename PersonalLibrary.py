import os
import json
import time
from datetime import datetime
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import openai
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

LIB_DIR = "library"
META_PATH = os.path.join(LIB_DIR, "library.json")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K = 5

os.makedirs(LIB_DIR, exist_ok=True)


def load_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"documents": [], "chunks": []}


def save_meta(meta):
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


meta = load_meta()

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text.append(t)
    return "\n\n".join(text)


def chunk_text(text: str):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return chunks


vectorizer = TfidfVectorizer()

def build_embeddings():
    texts = [c["text"] for c in meta["chunks"]]
    if texts:
        X = vectorizer.fit_transform(texts)
        return X
    return None


def add_pdf(uploaded_file, title=None):
    fname = uploaded_file.name
    now_ts = int(time.time())
    saved_name = f"{now_ts}_{fname}"
    save_path = os.path.join(LIB_DIR, saved_name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    text = extract_text_from_pdf(save_path)
    if not title:
        title = fname

    doc_id = len(meta["documents"]) + 1
    meta["documents"].append({
        "id": doc_id,
        "title": title,
        "path": save_path,
        "uploaded_at": datetime.utcnow().isoformat()
    })

    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        meta["chunks"].append({
            "doc_id": doc_id,
            "chunk_index": i,
            "text": chunk
        })

    save_meta(meta)
    st.success(f"Added {uploaded_file.name} to your library!")
    return doc_id, len(chunks)

def retrieve(query, top_k=TOP_K):
    X = build_embeddings()
    if X is None:
        return []
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, X)[0]
    top_idx = sims.argsort()[::-1][:top_k]
    results = [(meta["chunks"][i], sims[i]) for i in top_idx]
    return results

def answer_extractive(query, retrieved):
    if not retrieved:
        return "No relevant documents found."
    parts = []
    sources = []
    for row, score in retrieved:
        parts.append(row["text"][:1000])
        sources.append(f"doc:{row['doc_id']}#chunk:{row['chunk_index']}")
    return f"Extracted passages:\n\n" + "\n---\n".join(parts) + f"\n\nSources: {', '.join(sources)}"

def answer_with_openai(query, retrieved, model="gpt-4o-mini"):
    if not OPENAI_AVAILABLE:
        return "OpenAI not available."
    contexts = [row["text"] for row, _ in retrieved]
    context_blob = "\n\n---\n\n".join(contexts)
    system = "Answer the question using ONLY the provided context."
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "Set OPENAI_API_KEY environment variable."
    openai.api_key = api_key
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"CONTEXT:\n{context_blob}\n\nQUESTION: {query}"}
            ],
            max_tokens=400,
            temperature=0
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"OpenAI call failed: {e}"






st.set_page_config(page_title="RAGe-Lib", layout="wide")
st.title("RAGe-Lib")

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

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Add PDF")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    title_in = st.text_input("Optional title")
    if uploaded:
        doc_id, n_chunks = add_pdf(uploaded, title_in or None)
        st.success(f"Added doc {doc_id} with {n_chunks} chunks")
        st.experimental_rerun()

with col2:
    st.subheader("Ask a question")
    query = st.text_area("Enter question")
    if st.button("Search & Answer"):
        if not query.strip():
            st.warning("Enter a question.")
        else:
            results = retrieve(query, top_k)
            st.write("**Top passages:**")
            for row, score in results:
                st.markdown(f"- doc:{row['doc_id']} chunk:{row['chunk_index']} (score {score:.3f})")
                st.write(row["text"][:500] + ("..." if len(row["text"]) > 500 else ""))
            if use_openai:
                answer = answer_with_openai(query, results)
                st.markdown("### Answer (OpenAI)")
                st.write(answer)
            else:
                answer = answer_extractive(query, results)
                st.markdown("### Extractive Answer")
                st.write(answer)

st.markdown("---")
st.caption("RAG library. Stores PDFs & metadata under ./library. Backup if important.")
