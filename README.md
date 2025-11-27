DocuQA: Domain-Specific RAG System with Citations
Overview
DocuQA is an advanced AI Knowledge Assistant designed to answer questions based strictly on a curated set of internal documents (PDFs), eliminating the "hallucinations" common in standard LLMs.

Unlike generic chatbots, this system implements Retrieval-Augmented Generation (RAG) to index specialized data (e.g., Medical Guidelines, Legal Codes) and provides precise page-level citations for every answer, ensuring trust and verifiability.

✨ Key Features
Zero-Cost Tech Stack: Optimized to run using Free Tier architecture (Google Gemini + HuggingFace Local Embeddings).

Multi-Document Intelligence: capable of synthesizing answers across multiple large PDFs (tested on 300+ page medical journals).

Evidence-Based Answers: The "Citations Engine" lists the exact filename and page number for every claim made by the AI.

Vector Search: Utilizes high-dimensional semantic search (via Pinecone) to find relevant context even if keywords don't match exactly.

🛠️ Tech Stack
Orchestration: LlamaIndex (RAG Framework)

LLM: Google Gemini 1.5 Flash (Generation)

Embeddings: HuggingFace all-MiniLM-L6-v2 (Local, 384-dimensional)

Vector Database: Pinecone (Serverless Indexing)

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