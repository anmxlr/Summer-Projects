import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


PDF_FILE = "SOS2025.pdf"

loader = PyPDFLoader(PDF_FILE)
docs = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)


embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)


retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

api_key = os.getenv("OPENAI_API_KEY")
print(api_key)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=api_key)
embeddings = OpenAIEmbeddings(openai_api_key=api_key)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

query = "Summarize the main findings of this PDF."
result = qa_chain.invoke({"query": query})

print("Answer:")
print(result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print("-", doc.metadata.get("source", "unknown"))
