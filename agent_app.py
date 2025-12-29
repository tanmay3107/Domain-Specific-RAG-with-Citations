import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.gemini import Gemini 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore

# 1. SETUP
load_dotenv()

# --- FIX: USE SPECIFIC MODEL VERSION ---
Settings.llm = Gemini(model="models/gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))

# Use Local Embeddings (Free)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. CONNECT TO EXISTING PINECONE INDEX
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("medical-knowledge-base")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# 3. CREATE THE TOOL
medical_tool = QueryEngineTool(
    query_engine=index.as_query_engine(similarity_top_k=3),
    metadata=ToolMetadata(
        name="medical_guidelines",
        description="Useful for answering specific questions about Paracetamol, Tuberculosis, and medical dosages. Always check this for clinical facts."
    ),
)

# 4. CREATE THE AGENT (FIXED INITIALIZATION)
print("Initializing Agent...")

# Try the standard method first. If your version is very new, it might use .from_tools
try:
    # Attempt 1: Standard ReActAgent (Newer versions)
    agent = ReActAgent.from_tools([medical_tool], llm=Settings.llm, verbose=True)
except AttributeError:
    print("⚠️ 'from_tools' not found. Trying legacy initialization...")
    # Attempt 2: Direct Initialization (Older versions fallback)
    from llama_index.core.agent import AgentRunner
    from llama_index.core.agent import ReActAgentWorker
    
    step_engine = ReActAgentWorker.from_tools([medical_tool], llm=Settings.llm, verbose=True)
    agent = AgentRunner(step_engine)

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