import os
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
load_dotenv()
# ---------------- CONFIG ----------------
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None

# ---------------- INIT COMPONENTS ----------------
def initialize_components():
    global llm, vector_store

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize LLM
    if llm is None:
        llm = ChatGroq(
            model="llama3-70b-8192",  # valid Groq model
            temperature=0.3,
            max_tokens=500,
            groq_api_key=os.getenv("GROQ_API_KEY")  # must be set in Streamlit Secrets
        )

    # Initialize Vector Store
    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR)
        )

# ---------------- PROCESS URLS ----------------
def process_urls(urls):
    global vector_store

    yield "Initializing components..."
    initialize_components()

    # Delete old collection
    try:
        vector_store.delete_collection()
        yield "Old database cleared"
    except:
        yield "No previous database found"

    # Recreate fresh DB
    vector_store = None
    initialize_components()

    yield "Loading data from URLs..."
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()

    if not documents:
        yield "No content loaded from URLs"
        return

    yield "Splitting text..."
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )

    docs = splitter.split_documents(documents)
    yield f"Created {len(docs)} chunks"

    yield "Adding documents to vector database..."
    ids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(docs, ids=ids)

    yield "Vector database ready ✅"

# ---------------- QUERY ----------------
def generate_answer(query):
    global vector_store, llm

    if vector_store is None or llm is None:
        raise RuntimeError("Vector DB not initialized. Please process URLs first.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    result = chain.invoke({"question": query})

    return result["answer"], result["sources"]