import warnings
warnings.filterwarnings('ignore')

import os
import cv2
import pytesseract
from PIL import Image
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update if needed

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API keys
if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing API keys in .env file")

# Initialize Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "rag-index"
DIMENSION = 1536

# Delete and recreate index
try:
    if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
        print(f"Deleting existing index: {INDEX_NAME}")
        pinecone.delete_index(INDEX_NAME)
except Exception as e:
    print(f"Error deleting index: {e}")

try:
    print(f"Creating index: {INDEX_NAME}")
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    print("Index created successfully")
except Exception as e:
    print(f"Error creating index: {e}")
    raise

index = pinecone.Index(INDEX_NAME)

# Load and OCR the image
def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        print("OCR text extracted successfully.")
        return text.strip()
    except Exception as e:
        print(f"Error reading image: {e}")
        return ""

# Prepare DataFrame from OCR text
def load_data_from_image(image_path):
    text = extract_text_from_image(image_path)
    if not text:
        raise ValueError("No text extracted from image.")
    data = {
        'id': ['img_doc_1'],
        'text': [text],
        'metadata': [{'source': 'image', 'category': 'extracted'}]
    }
    return pd.DataFrame(data)

image_path = r"C:\Users\Acer\Desktop\AI_AGENT\Pinecone\AIML_2021.jpg"
df = load_data_from_image(image_path)

# Embedding generation and storage
def generate_and_store_embeddings(df, index, batch_size=1):
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    for i in tqdm(range(0, len(df), batch_size), desc="Processing embeddings"):
        batch = df.iloc[i:i+batch_size]
        texts = batch['text'].tolist()
        embeddings = openai_client.embeddings.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        vectors = []
        for idx, row in batch.iterrows():
            vectors.append({
                'id': row['id'],
                'values': embeddings.data[idx].embedding,
                'metadata': {
                    'text': row['text'],
                    **row.get('metadata', {})
                }
            })
        index.upsert(vectors=vectors)
    print("Embeddings stored successfully.")
    print(f"Index stats: {index.describe_index_stats()}")

generate_and_store_embeddings(df, index)

# Query Pinecone
def rag_query(query, index, top_k=3):
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    embed = openai_client.embeddings.create(
        input=[query],
        model="text-embedding-ada-002"
    ).data[0].embedding
    res = index.query(vector=embed, top_k=top_k, include_metadata=True)
    return [match['metadata']['text'] for match in res.matches]

# Generate answer
def generate_response(query, context):
    if not context:
        return "No relevant context found."
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    context_block = "\n".join([f"{i+1}. {ctx}" for i, ctx in enumerate(context)])
    prompt = f"""Answer the question based on the context below. Be concise and accurate.
    
Context:
{context_block}

Question: {query}
Answer:"""
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

# Interactive loop
if __name__ == "__main__":
    print("\nRAG System Ready (Image-based) — type 'quit' to exit")
    while True:
        query = input("\nEnter your question: ").strip()
        if query.lower() in ['quit', 'exit']:
            break
        if not query:
            continue
        print("\nSearching for relevant information...")
        context = rag_query(query, index)
        if context:
            print("\nTop matching context:")
            for i, ctx in enumerate(context, 1):
                print(f"\nContext {i}:\n{ctx[:500]}...\n" if len(ctx) > 500 else ctx)
        else:
            print("No relevant context found.")
            continue
        print("\nGenerating response...")
        response = generate_response(query, context)
        print("\nResponse:")
        print(response)
