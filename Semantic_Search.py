# Lesson 1 - Semantic Search - Fixed Version

# Import the Needed Packages
import warnings
warnings.filterwarnings('ignore')

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
import time
import torch
from tqdm.auto import tqdm
import json
from datetime import datetime

# Load the Dataset
dataset = load_dataset('quora', split='train[240000:290000]')

# Extract questions - Corrected access pattern
questions = []
for record in dataset:
    questions.extend(record['questions']['text'])  # Fixed nested structure

# Remove duplicates
questions = list(set(questions))

# Show sample and count
print('\n'.join(questions[:10]))
print('-' * 50)
print(f'Number of questions: {len(questions)}')

# Check CUDA and Setup the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    print('Warning: No CUDA GPU available, using CPU.')

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Encode a sample query
query = 'which city is the most populated in the world?'
xq = model.encode(query)
print("Query vector shape:", xq.shape)

# Setup Pinecone - Modified for free tier compatibility
def get_pinecone_api_key():
    # Option 1: Get from environment variable (recommended)
    # return os.environ.get('PINECONE_API_KEY')
    
    # Option 2: Replace with your actual API key (but don't commit this!)
    return "pcsk_34bGGf_QubPBPzzsH4LrjCXea9J5dtyQ8X9LgqFjpFy3AWaeAkvLEZibjE3EK1MGuBPZf9"  # <<< REPLACE WITH YOUR KEY

def create_index_name(base_name):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{base_name}-{timestamp}"

PINECONE_API_KEY = get_pinecone_api_key()
pinecone = Pinecone(api_key=PINECONE_API_KEY)

# Generate a unique index name
INDEX_NAME = create_index_name('quora-search')

# If index exists, delete it
existing_indexes = pinecone.list_indexes()
if INDEX_NAME in [index.name for index in existing_indexes]:
    pinecone.delete_index(INDEX_NAME)

print("Creating index:", INDEX_NAME)

# Create a new index - Using free tier compatible region
try:
    pinecone.create_index(
        name=INDEX_NAME, 
        dimension=model.get_sentence_embedding_dimension(),  # Fixed typo in method name
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'  # Changed to free-tier compatible region
        )
    )
except Exception as e:
    print(f"Error creating index: {e}")
    print("\nNote: If you're on Pinecone's free tier, you might need to:")
    print("1. Use 'us-east-1' as your region")
    print("2. Upgrade your plan if you need other regions")
    exit(1)

# Connect to the index
index = pinecone.Index(INDEX_NAME)
print("Connected to index:", index)

# Create Embeddings and Upsert to Pinecone
batch_size = 200
vector_limit = 10000  # Reduced for free tier compatibility
questions = questions[:vector_limit]

for i in tqdm(range(0, len(questions), batch_size)):
    try:
        # Find end of batch
        i_end = min(i+batch_size, len(questions))
        # Create IDs batch
        ids = [str(x) for x in range(i, i_end)]
        # Create metadata batch
        metadatas = [{'text': text} for text in questions[i:i_end]]
        # Create embeddings
        xc = model.encode(questions[i:i_end])
        # Create records list for upsert
        records = zip(ids, xc, metadatas)
        # Upsert to Pinecone
        index.upsert(vectors=records)
    except Exception as e:
        print(f"Error processing batch {i}-{i_end}: {e}")
        continue

# View index stats
try:
    print(index.describe_index_stats())
except Exception as e:
    print(f"Error getting index stats: {e}")

# Run Your Query
def run_query(query):
    try:
        embedding = model.encode(query).tolist()
        results = index.query(
            top_k=10,
            vector=embedding,
            include_metadata=True,
            include_values=False
        )
        print(f"\nResults for query: '{query}'")
        for result in results['matches']:
            print(f"{round(result['score'], 2)}: {result['metadata']['text']}")
    except Exception as e:
        print(f"Error running query: {e}")

# Example queries
run_query('which city has the highest population in the world?')
run_query('how do i make chocolate cake?')

print("\nScript completed successfully!")

import pandas as pd

# Save as CSV
df = pd.DataFrame(questions, columns=['text'])
df.to_csv('quora_questions.csv', index=False)

# Or as JSON
with open('quora_questions.json', 'w') as f:
    json.dump(questions, f)