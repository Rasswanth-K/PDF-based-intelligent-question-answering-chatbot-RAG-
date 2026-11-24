from dotenv import load_dotenv
import os
import logging
import streamlit as st
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader

# --- Environment & Logging Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RAG-Senior-Fix")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("❌ CRITICAL: GEMINI_API_KEY missing in environment variables or .env file.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# --- Configure Hybrid Models ---
Settings.llm = GoogleGenAI(model="gemini-3.6-flash", api_key=api_key, temperature=0.1)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# --- UI Configuration ---
st.set_page_config(page_title="Enterprise PDF RAG", page_icon="📄", layout="centered")
st.title("📄 Enterprise PDF RAG (PyMuPDF Fix + Quota Handler)")

# --- Ingestion & Persistence Engine with Robust PDF Parsing ---
@st.cache_resource
def initialize_rag_system(
    pdf_path: str = "iso27001.pdf", 
    persist_dir: str = "./chroma_clean_metadata_v3",
    collection_name: str = "enterprise_pdf_v3"
):
    try:
        if not os.path.exists(pdf_path):
            st.error(f"❌ Source document '{pdf_path}' not found in root directory.")
            st.stop()

        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = chroma_client.get_or_create_collection(collection_name)
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if chroma_collection.count() > 0:
            logger.info(f"Loaded existing index. Collection count: {chroma_collection.count()} vectors.")
            return VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context,
            )

        with st.spinner("Parsing 450+ pages cleanly with PyMuPDF, normalizing metadata, and embedding..."):
            # Use PyMuPDFReader to bypass raw binary/FlateDecode stream corruption
            loader = PyMuPDFReader()
            documents = loader.load(file_path=pdf_path)
            
            # Explicitly stamp sequential page/doc metadata for fallback tracking
            for idx, doc in enumerate(documents):
                raw_page = doc.metadata.get("page_label") or doc.metadata.get("page")
                doc.metadata["display_page"] = str(raw_page) if raw_page else str(idx + 1)

            splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                transformations=[splitter],
                show_progress=True,
            )
            logger.info("Vector index successfully committed with clean text chunks.")
            return index

    except Exception as e:
        logger.error(f"Ingestion/Initialization failure: {str(e)}", exc_info=True)
        st.error(f"⚠️ Pipeline initialization error: {str(e)}")
        st.stop()

# --- Query Engine Setup ---
dl_index = initialize_rag_system()
query_engine = dl_index.as_query_engine(
    similarity_top_k=8,
    response_mode="compact"
)

# --- Chat Interface State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Query your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Retrieving & synthesizing..."):
        try:
            response = query_engine.query(prompt)
            bot_reply = str(response)
            
            retrieved_chunks = []
            cited_pages = set()
            for node in getattr(response, "source_nodes", []):
                meta = node.node.metadata
                page_num = meta.get("display_page") or meta.get("page_label") or meta.get("page") or "Index-bound"
                cited_pages.add(page_num)
                snippet = node.node.text[:300].replace("\n", " ")
                score = getattr(node, "score", "N/A")
                retrieved_chunks.append(f"**Ref Page/Node {page_num}** (Score: `{score}`):\n> {snippet}...")

            if cited_pages:
                sorted_pages = sorted(list(cited_pages), key=lambda x: int(x) if str(x).isdigit() else 9999)
                bot_reply += f"\n\n--- *Source References*: Pages/Nodes {', '.join(map(str, sorted_pages[:8]))}"

        except Exception as e:
            err_str = str(e)
            logger.error(f"Query execution error: {err_str}", exc_info=True)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                bot_reply = "🚨 **Gemini Free-Tier Daily Quota Exceeded (20 req/day limit reached)**. Wait 24 hours or switch LLM backend to Ollama (local) to bypass limits."
            else:
                bot_reply = f"⚠️ Error executing query: {err_str}"
            retrieved_chunks = []

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
        if retrieved_chunks:
            with st.expander("🔍 Debug: View Retrieved Chunks (Top-8 Context)"):
                for chunk in retrieved_chunks:
                    st.markdown(chunk)
                    st.divider()