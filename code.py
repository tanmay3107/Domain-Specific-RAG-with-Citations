import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec # Import ServerlessSpec for creation
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore

# 1. SECURITY SETUP
load_dotenv()

if not os.getenv("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY not found. Check your .env file.")

# 2. INITIALIZE PINECONE
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "medical-bot-index"

# --- FIX START: AUTOMATIC INDEX CREATION ---
# Check if index exists. If not, create it.
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"Index '{index_name}' not found. Creating it now...")
    try:
        pc.create_index(
            name=index_name,
            dimension=1536, # 1536 is the dimension for OpenAI's default embedding model
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1" # The free tier is usually in us-east-1
            )
        )
        print("Index creating... waiting for it to be ready.")
        time.sleep(10) # Give Pinecone a moment to initialize
    except Exception as e:
        print(f"Failed to create index: {e}")
        exit()
else:
    print(f"Index '{index_name}' already exists. Connecting...")
# --- FIX END ---

# Connect to the index
pinecone_index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# 3. INGESTION
print("Loading documents from ./medical_pdfs...")
# ERROR HANDLING: Check if folder exists
if not os.path.exists("./medical_pdfs"):
    os.makedirs("./medical_pdfs")
    print("Created ./medical_pdfs folder. Please put a PDF in there and run again!")
    exit()

documents = SimpleDirectoryReader("./medical_pdfs").load_data()

if not documents:
    print("No PDFs found in ./medical_pdfs. Please add a file.")
    exit()

storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("Generating embeddings and uploading to Pinecone...")
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# 4. RETRIEVAL ENGINE
query_engine = index.as_query_engine(similarity_top_k=3)

def ask_bot(question):
    print(f"\nAsking: {question}...")
    response = query_engine.query(question)
    
    print(f"**Answer:** {response}\n")
    print("--- Citations (Evidence) ---")
    
    for node in response.source_nodes:
        page = node.node.metadata.get('page_label', 'N/A')
        doc_name = node.node.metadata.get('file_name', 'Unknown')
        print(f"Found in: {doc_name} | Page: {page}")
        clean_text = node.node.get_text().replace("\n", " ")[:150]
        print(f"Excerpt: {clean_text}...\n")

# Usage
if __name__ == "__main__":
    ask_bot("What are the side effects of Paracetamol?")