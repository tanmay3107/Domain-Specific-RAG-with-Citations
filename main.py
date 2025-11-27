import os
import sys
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini 

# 1. SETUP
load_dotenv()

# Configure LLM (Gemini) and Embeddings (HuggingFace)
Settings.llm = Gemini(model="models/gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. PINECONE CONNECTION
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "medical-knowledge-base" 

# Check if we need to create the index
existing_indexes = [i["name"] for i in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"Creating new index: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=384, # Matching local embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(10) # Wait for initialization

pinecone_index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# 3. INGESTION (The Heavy Lifting)
# This reads EVERY PDF in the folder
print("scanning ./medical_pdfs folder...")
documents = SimpleDirectoryReader("./medical_pdfs").load_data()
print(f"Found {len(documents)} pages of medical data.")

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Note: This step might take 1-2 minutes 
# It converts text -> numbers -> Pinecone
print("Updating Knowledge Base (this may take time for large files)...")
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
print("✅ Knowledge Base Ready!")

# 4. INTERACTIVE CHAT LOOP
query_engine = index.as_query_engine(similarity_top_k=5) # Checks top 5 relevant pages

def start_chat():
    print("\n" + "="*50)
    print("🤖 AI MEDICAL ASSISTANT READY")
    print("Type 'exit' or 'q' to quit.")
    print("="*50 + "\n")

    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
            
        print("Thinking...")
        try:
            response = query_engine.query(user_input)
            
            # Print the AI Answer
            print(f"\nAI: {response}\n")
            
            # Print the Sources (The "Strong Profile" feature)
            print("-" * 30)
            print("Sources Used:")
            seen_sources = set()
            for node in response.source_nodes:
                file_name = node.node.metadata.get('file_name', 'Unknown')
                page_label = node.node.metadata.get('page_label', 'N/A')
                
                # Avoid printing the same page twice
                source_id = f"{file_name} (Page {page_label})"
                if source_id not in seen_sources:
                    print(f"• {source_id}")
                    seen_sources.add(source_id)
            print("-" * 30 + "\n")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    start_chat()