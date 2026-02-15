import os
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

# LangChain Imports
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# ---------------- CONFIG ----------------
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


def initialize_components():
    global llm, vector_store

    # Create directory if it doesn't exist
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=500
            # groq_api_key is handled by os.environ/load_dotenv
        )

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )


def process_urls(urls):
    """
    Scrapes data from URLs and stores it in a vector db.
    """
    # --- FIX: Declare global to modify the global variable ---
    global vector_store
    # ---------------------------------------------------------

    yield "Initializing Components..."
    initialize_components()

    yield "Resetting vector store...✅"
    try:
        # We try to clear the old collection.
        # If vector_store is not fully init, this might fail, hence the try/except
        if vector_store:
            vector_store.delete_collection()
            vector_store = None  # Force re-initialization
            initialize_components()
    except Exception as e:
        yield f"⚠️ Note: DB reset skipped ({e})"
        pass

    yield "Loading data...✅"

    try:
        # WebBaseLoader is robust and handles User-Agents better than Unstructured
        loader = WebBaseLoader(urls)
        data = loader.load()
    except Exception as e:
        yield f"❌ Error loading data: {str(e)}"
        return

    if not data:
        yield "⚠️ Warning: No data loaded."
        return

    yield f"Loaded {len(data)} documents. Splitting text...✅"

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(data)

    yield f"Created {len(docs)} chunks. Adding to vector database...✅"

    if docs:
        uuids = [str(uuid4()) for _ in range(len(docs))]
        vector_store.add_documents(docs, ids=uuids)

    yield "Done! Vector database is ready. ✅"


def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is not initialized. Run process_urls first.")

    # Using 'stuff' chain to feed context to LLM
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3})
    )

    result = chain.invoke({"question": query})

    return result['answer'], result['sources']


if __name__ == "__main__":
    # 1. Check for API Key
    if not os.getenv("GROQ_API_KEY"):
        print("❌ ERROR: GROQ_API_KEY not found. Please set it in your .env file.")
        exit(1)

    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    print("--- Starting Processing ---")

    # 2. Iterate generator to execute logic
    for status in process_urls(urls):
        print(status)

    print("--- Processing Complete ---")

    # 3. Generate Answer
    query = "Tell me what was the 30 year fixed mortgage rate along with the date?"
    try:
        answer, sources = generate_answer(query)
        print("\n=================================")
        print(f"Question: {query}")
        print(f"Answer: {answer}")
        print(f"Sources: {sources}")
        print("=================================")
    except Exception as e:
        print(f"An error occurred: {e}")