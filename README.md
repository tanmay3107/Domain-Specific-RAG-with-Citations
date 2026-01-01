Agentic AI: Multi-Agent RAG with Tool Calling

Overview

This project implements an Agentic RAG (Retrieval-Augmented Generation) system designed to answer complex domain-specific questions. Unlike standard RAG chatbots that strictly retrieve data, this system uses an Agentic Workflow where the LLM reasons about the user's query and dynamically decides when to use external tools.

It features a Reasoning Loop that utilizes a Vector Database (Pinecone) as a "Knowledge Tool," allowing the agent to fetch clinical guidelines only when necessary and cross-reference answers with citations.

✨ Key Features

Agentic Workflow: Built using LlamaIndex agents that break down queries and execute multi-step reasoning.

Tool Calling Capability: The system wraps the Vector Store as a QueryEngineTool, enabling the LLM to "call" the database like an API function.

Citation & Grounding: The agent provides exact page-level citations from PDF documents, reducing hallucinations.

Vector Search: Uses Pinecone for high-precision semantic retrieval of medical context.

🛠️ Tech Stack

Orchestration: LlamaIndex (ReAct Agent)

LLM: Google Gemini 1.5 Flash

Vector DB: Pinecone (Serverless)

Embeddings: HuggingFace all-MiniLM-L6-v2 (Local)

Language: Python 3.10+

⚙️ System Architecture
Ingestion: PDFs are loaded and split into manageable chunks.

Embedding: Text chunks are converted into vector embeddings locally using HuggingFace.

Indexing: Vectors are stored in the Pinecone cloud database.

Retrieval: User queries are converted to vectors; the system performs a cosine-similarity search to find the Top-5 relevant pages.

Synthesis: retrieved context is fed to Gemini 1.5 Flash to generate a natural language response with source metadata.

🚀 How to Run Locally
1. Clone the Repository

Bash

git clone https://github.com/your-username/docuqa-rag.git
cd docuqa-rag
2. Install Dependencies

Bash

pip install -r requirements.txt
3. Set Up Environment Variables Create a .env file in the root directory:

Code snippet

PINECONE_API_KEY=your_pinecone_key
GOOGLE_API_KEY=your_google_gemini_key
4. Add Documents Place your PDF files (e.g., Medical Journals, Textbooks) inside the medical_pdfs/ folder.

5. Run the Application

Bash

python main.py
### 📸 Example Output
<img width="1263" height="366" alt="RAG System Demo Screenshot" src="https://github.com/user-attachments/assets/f1415ec2-6328-41bf-b43a-c85bb98dd26c" />

*Screenshot: The AI correctly retrieving information about Paracetamol side effects and citing the specific PDF source.*

🔮 Future Improvements
[ ] Web Interface: Deploying a Streamlit frontend for a user-friendly chat experience.

[ ] Hybrid Search: Combining Keyword search (BM25) with Vector search for better accuracy on specific technical terms.

[ ] Chat History: Adding memory so the bot remembers previous questions in the session.