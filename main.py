import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI # <--- UPDATED IMPORT

# 1. SETUP & CONFIGURATION
load_dotenv()

# Verify Keys
if not os.getenv("PINECONE_API_KEY"):
    raise ValueError("Missing PINECONE_API_KEY in .env")
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Missing GOOGLE_API_KEY in .env")

# --- FREE TECH STACK CONFIGURATION ---
print("Setting up Free Local Embeddings & Gemini LLM...")

# A. Set Embed Model (Runs on your CPU/GPU - Free)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# B. Set LLM (Google Gemini - Free Tier) - UPDATED CODE
Settings.llm = GoogleGenAI(
    model="models/gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# 2. PINECONE INDEX MANAGEMENT
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "medical-bot-index"

# Check if index exists and verify dimensions
existing_indexes = [i["name"] for i in pc.list_indexes()]

if index_name in existing_indexes:
    # Check dimensions (Local embeddings use 384, not 1536)
    index_info = pc.describe_index(index_name)
    if index_info.dimension != 384:
        print(f"⚠️  Old index has wrong dimensions ({index_info.dimension}). Deleting to recreate with 384...")
        pc.delete_index(index_name)
        time.sleep(5) 
        existing_indexes.remove(index_name)

# Create Index if it doesn't exist
if index_name not in existing_indexes:
    print(f"Creating new index '{index_name}' with dimension 384...")
    pc.create_index(
        name=index_name,
        dimension=384, 
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(10) 

# Connect
pinecone_index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# 3. INGESTION
print("Loading documents...")
documents = SimpleDirectoryReader("./medical_pdfs").load_data()

if not documents:
    print("❌ No files found in ./medical_pdfs")
    exit()

storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("Generating embeddings (Locally)...")
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# 4. QUERY
query_engine = index.as_query_engine(similarity_top_k=3)

def ask_bot(question):
    print(f"\nAsking: {question}...")
    response = query_engine.query(question)
    
    print(f"\n**Answer:** {response}\n")
    print("--- Citations ---")
    for node in response.source_nodes:
        doc = node.node.metadata.get('file_name', 'Unknown')
        page = node.node.metadata.get('page_label', 'N/A')
        print(f"Source: {doc} (Page {page})")

if __name__ == "__main__":
    ask_bot("What are the precautions for Paracetamol?")