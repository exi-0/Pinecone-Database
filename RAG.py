import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
import os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

# Get API keys from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API keys
if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing API keys in .env file")

# Initialize Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY)

# Index configuration
INDEX_NAME = "rag-index"
DIMENSION = 1536  # Dimension for text-embedding-ada-002

# Delete index if it exists
try:
    existing_indexes = pinecone.list_indexes()
    if INDEX_NAME in [index.name for index in existing_indexes]:
        print(f"Deleting existing index: {INDEX_NAME}")
        pinecone.delete_index(INDEX_NAME)
except Exception as e:
    print(f"Error deleting index: {e}")

# Create new index with free-tier compatible region
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

# Improved data loading with better dummy data
def load_data(file_path='./data/wiki.csv', max_articles=100):
    try:
        df = pd.read_csv(file_path, nrows=max_articles)
        print(f"Loaded {len(df)} articles from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Generating enhanced dummy data...")
        
        # Create more meaningful dummy data
        sample_articles = [
            {
                'id': f"doc_{i}",
                'text': f"Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. {i}",
                'metadata': {'source': 'dummy', 'category': 'technology'}
            } if i % 3 == 0 else {
                'id': f"doc_{i}",
                'text': f"YouTube is an online video-sharing platform where users can upload, view, and share videos. {i}",
                'metadata': {'source': 'dummy', 'category': 'media'}
            } if i % 3 == 1 else {
                'id': f"doc_{i}",
                'text': f"Machine learning is a subset of AI that focuses on building systems that learn from data. {i}",
                'metadata': {'source': 'dummy', 'category': 'technology'}
            }
            for i in range(max_articles)
        ]
        
        df = pd.DataFrame(sample_articles)
        return df

df = load_data(max_articles=50)  # Reduced number for demo

# Enhanced embedding generation
def generate_and_store_embeddings(df, index, batch_size=50):
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        for i in tqdm(range(0, len(df), batch_size), desc="Processing embeddings"):
            batch = df.iloc[i:i+batch_size]
            
            # Generate embeddings for text
            texts = batch['text'].tolist()
            embeddings = openai_client.embeddings.create(
                input=texts,
                model="text-embedding-ada-002"
            )
            
            # Prepare vectors with proper metadata
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
            
            # Upsert to Pinecone
            index.upsert(vectors=vectors)
        
        print("Embeddings stored successfully")
        print(f"Index stats: {index.describe_index_stats()}")
    except Exception as e:
        print(f"Error in embedding generation: {e}")
        raise

generate_and_store_embeddings(df, index)

# Improved query function
def rag_query(query, index, top_k=3):
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Get query embedding
        embed = openai_client.embeddings.create(
            input=[query],
            model="text-embedding-ada-002"
        ).data[0].embedding
        
        # Query Pinecone with proper error handling
        res = index.query(
            vector=embed,
            top_k=top_k,
            include_metadata=True
        )
        
        return [match['metadata']['text'] for match in res.matches]
    except Exception as e:
        print(f"Error in query: {str(e)}")
        return []

# Enhanced response generation
def generate_response(query, context):
    try:
        if not context:
            return "I couldn't find enough relevant information to answer that question."
            
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Build context string separately to avoid f-string issues
        context_items = [f"{i+1}. {ctx}" for i, ctx in enumerate(context)]
        context_block = "\n".join(context_items)
        
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
    except Exception as e:
        print(f"Error in response generation: {e}")
        return "I encountered an error while generating a response."

# Interactive query loop
if __name__ == "__main__":
    print("\nRAG System Ready (type 'quit' to exit)")
    while True:
        try:
            query = input("\nEnter your question: ").strip()
            if query.lower() in ['quit', 'exit']:
                break
            if not query:
                continue
                
            print("\nSearching for relevant information...")
            context = rag_query(query, index, top_k=3)
            
            if context:
                print("\nTop matching contexts found:")
                for i, ctx in enumerate(context[:3], 1):
                    print(f"\nContext {i}:")
                    print(ctx[:500] + "..." if len(ctx) > 500 else ctx)
            else:
                print("No relevant context found")
                continue
                
            print("\nGenerating response...")
            response = generate_response(query, context)
            print("\nResponse:")
            print(response)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")