Pinecone Vector Database Projects
This repository showcases various AI/ML use cases powered by Pinecone, a high-performance vector database, to enable semantic search, recommender systems, hybrid retrieval, facial similarity search, and more.

🗂️ Overview
Pinecone is a managed vector database optimized for similarity search. This project demonstrates how to integrate Pinecone with AI/ML models to perform:

🔍 Semantic & hybrid search

🎯 Recommendation engines

🧠 Facial embedding comparison

⚠️ Anomaly detection

🧾 Image text extraction

🤖 Retrieval-Augmented Generation (RAG)

📁 Project Structure
bash
Copy
Edit
.
├── Anomaly_Detection.py            # Detect unusual data points using embeddings
├── Facial_Similarity_Search.py    # Compare face embeddings for similarity
├── Hybrid_Search.py               # Combine dense + sparse search with Pinecone
├── image_extract_info.py          # Extract and embed image metadata/text
├── RAG.py                         # Retrieval-Augmented Generation using Pinecone
├── Recommender_System.py          # Recommendation engine with vector similarity
├── Semantic_Search.py             # Semantic retrieval using SentenceTransformers
├── quora_questions.json           # Sample dataset (Quora questions)
├── vectors.vec                    # Sample saved vectors (optional binary format)
├── AIML_2021.jpg                  # Sample image input
├── test.py / hs.py / tempCode... # Utility or experimental scripts
└── requirements.txt               # Python dependencies
🔧 Setup Instructions
Clone the repository

bash
Copy
Edit
git clone https://github.com/yourusername/Pinecone-Database.git
cd Pinecone-Database
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set up API keys

Create a .env file in the root directory and add:

ini
Copy
Edit
PINECONE_API_KEY=your-pinecone-key
OPENAI_API_KEY=your-openai-key
Run an example

bash
Copy
Edit
python Semantic_Search.py
🚀 Use Case Highlights
File	Purpose
Semantic_Search.py	Perform similarity search with sentence embeddings
Hybrid_Search.py	Use sparse (BM25) + dense embeddings together
Facial_Similarity_Search.py	Compare faces using embedding vectors
Recommender_System.py	Recommend items/documents based on similarity
RAG.py	Answer queries using Pinecone + LLM (OpenAI)
Anomaly_Detection.py	Identify outliers using vector space distance
image_extract_info.py	Use OCR/text embeddings from images

🧪 Tools & Libraries
🧠 Pinecone

🤖 OpenAI API

🔤 SentenceTransformers

📸 Pillow, tesseract, or easyocr for image text extraction

🧰 scikit-learn, numpy, pandas, torch for ML workflows

📜 License
MIT License – Use and modify freely for research or commercial applications.
