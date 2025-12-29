import os
import sys
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore

# 1. SETUP
load_dotenv()

# --- USE SPECIFIC MODEL VERSION ---
Settings.llm = Gemini(model="models/gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. CONNECT TO EXISTING PINECONE INDEX
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("medical-knowledge-base")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# 3. CREATE THE AGENT (The Stable Way)
print("Initializing Agent...")

try:
    # "chat_mode='react'" forces it to behave like an Agent that uses tools.
    # It treats the Vector Store Index as a lookup tool automatically.
    agent = index.as_chat_engine(
        chat_mode="react", 
        verbose=True
    )
    print("✅ Agent initialized successfully (ReAct Mode)!")

except Exception as e:
    print(f"⚠️ ReAct mode failed: {e}")
    print("Falling back to standard Context Chat...")
    # Fallback: Standard RAG Chat (Still smart, just less 'agentic' internals)
    agent = index.as_chat_engine(chat_mode="context", verbose=True)

# 5. CHAT LOOP
print("\n🤖 MEDICAL AGENT READY (Type 'q' to quit)")
print("Try asking: 'What is the dosage for Paracetamol?' (Watch it use the tool!)")
print("Try asking: 'Hello there' (Watch it NOT use the tool)")
print("-" * 50)

while True:
    try:
        user_input = input("You: ")
        if user_input.lower() in ['q', 'exit']: break
        
        response = agent.chat(user_input)
        print(f"Agent: {response}\n")
    except Exception as e:
        print(f"❌ Error: {e}")