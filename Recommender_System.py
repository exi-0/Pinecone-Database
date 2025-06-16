import os
import pandas as pd
import warnings
from tqdm.auto import tqdm
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()
warnings.filterwarnings('ignore')

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize OpenAI and Pinecone clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pinecone = Pinecone(api_key=PINECONE_API_KEY)

# Define Pinecone index name
INDEX_NAME = "dl-ai-recommender"

# Delete existing index if it exists
existing_indexes = [index.name for index in pinecone.list_indexes()]
if INDEX_NAME in existing_indexes:
    print(f"Deleting existing index '{INDEX_NAME}'...")
    pinecone.delete_index(INDEX_NAME)

# Create a new index with free-tier compatible region (AWS us-east-1)
print(f"Creating new index '{INDEX_NAME}' in free-tier compatible region...")
pinecone.create_index(
    name=INDEX_NAME,
    dimension=1536,
    metric='cosine',
    spec=ServerlessSpec(cloud='aws', region='us-east-1')  # free-tier compatible region
)

index = pinecone.Index(INDEX_NAME)

# Function to get embeddings from OpenAI
def get_embeddings(texts, model="text-embedding-ada-002"):
    response = openai_client.embeddings.create(input=texts, model=model)
    return response

# Load dataset (with error handling)
DATA_PATH = 'all-the-news-3.csv'
try:
    df = pd.read_csv(DATA_PATH, nrows=10000)  # Load up to 10k rows
except FileNotFoundError:
    raise FileNotFoundError(f"CSV file not found at '{DATA_PATH}'. Please ensure the file exists.")

# Embed article titles in chunks and upsert to Pinecone
CHUNK_SIZE = 400
progress_bar = tqdm(total=len(df), desc="Embedding titles")

prepped = []
for i in range(0, len(df), CHUNK_SIZE):
    chunk = df.iloc[i:i + CHUNK_SIZE]
    titles = chunk['title'].tolist()

    try:
        embeddings_response = get_embeddings(titles)
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        continue

    prepped.extend([
        {
            'id': str(i + idx),
            'values': embeddings_response.data[idx].embedding,
            'metadata': {'title': titles[idx]}
        }
        for idx in range(len(titles))
    ])

    # Upsert in batches of 200
    if len(prepped) >= 200:
        try:
            index.upsert(prepped)
            prepped = []
        except Exception as e:
            print(f"Error upserting to Pinecone: {e}")

    progress_bar.update(len(chunk))

# Upsert any remaining embeddings
if prepped:
    try:
        index.upsert(prepped)
    except Exception as e:
        print(f"Error upserting remaining items: {e}")

progress_bar.close()

print("\nIndex stats after title embeddings:")
print(index.describe_index_stats())

# Function to get recommendations based on a search term
def get_recommendations(pinecone_index, search_term, top_k=10):
    try:
        embed_resp = get_embeddings([search_term])
        embed = embed_resp.data[0].embedding
        query_response = pinecone_index.query(vector=embed, top_k=top_k, include_metadata=True)
        return query_response
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return None

# Example recommendations by title keyword
print("\nRecommendations based on keyword 'obama':")
recommendations = get_recommendations(index, 'obama')
if recommendations:
    for match in recommendations.matches:
        print(f"{match.score:.4f} : {match.metadata['title']}")

# Delete index to recreate it for full article embeddings
print(f"\nRecreating index '{INDEX_NAME}' for full article embeddings...")
if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
    pinecone.delete_index(INDEX_NAME)

pinecone.create_index(
    name=INDEX_NAME,
    dimension=1536,
    metric='cosine',
    spec=ServerlessSpec(cloud='aws', region='us-east-1')  # free-tier compatible region
)

articles_index = pinecone.Index(INDEX_NAME)

# Function to embed and upsert full articles
def embed_articles(embeddings_response, title, prepped_list, embed_id_start):
    embed_id = embed_id_start
    for embedding_obj in embeddings_response.data:
        prepped_list.append({
            'id': str(embed_id),
            'values': embedding_obj.embedding,
            'metadata': {'title': title}
        })
        embed_id += 1
        # Upsert in batches of 100
        if len(prepped_list) >= 100:
            try:
                articles_index.upsert(prepped_list)
                prepped_list.clear()
            except Exception as e:
                print(f"Error upserting articles: {e}")
    return embed_id

# Prepare for embedding full articles
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
prepped = []
embed_num = 0

# Load articles and titles from CSV
try:
    df_articles = pd.read_csv(DATA_PATH, nrows=100)
    articles_list = df_articles['article'].tolist()
    titles_list = df_articles['title'].tolist()
except Exception as e:
    print(f"Error loading articles: {e}")
    exit(1)

print("\nEmbedding full articles...")
for i, article_text in enumerate(tqdm(articles_list, desc="Processing articles")):
    title = titles_list[i]
    if article_text and isinstance(article_text, str):
        try:
            chunks = text_splitter.split_text(article_text)
            embeddings_response = get_embeddings(chunks)
            embed_num = embed_articles(embeddings_response, title, prepped, embed_num)
        except Exception as e:
            print(f"Error processing article {i}: {e}")

# Upsert any remaining article embeddings
if prepped:
    try:
        articles_index.upsert(prepped)
    except Exception as e:
        print(f"Error upserting remaining articles: {e}")

print("\nIndex stats after full article embeddings:")
print(articles_index.describe_index_stats())

# Final recommendations based on full article embeddings
print("\nRecommendations based on full article content for 'obama':")
final_reco = get_recommendations(articles_index, 'obama', top_k=10)
if final_reco:
    seen_titles = set()
    for match in final_reco.matches:
        title = match.metadata.get('title', 'Unknown Title')
        if title not in seen_titles:
            print(f"{match.score:.4f} : {title}")
            seen_titles.add(title)
