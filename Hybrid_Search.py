import warnings
warnings.filterwarnings('ignore')

import os
from dotenv import load_dotenv
from datasets import load_dataset
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import torch
from IPython.core.display import HTML
from io import BytesIO
from base64 import b64encode
import pandas as pd

# Load .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Setup Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "fashion-hybrid-index"

# Delete index if exists
if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
    pinecone.delete_index(INDEX_NAME)

# Create new index
pinecone.create_index(
    INDEX_NAME,
    dimension=512,
    metric="dotproduct",
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
)

index = pinecone.Index(INDEX_NAME)

# Load dataset
fashion = load_dataset("ashraq/fashion-product-images-small", split="train")
images = fashion['image']
metadata = fashion.remove_columns('image').to_pandas()

# BM25 encoder
bm25 = BM25Encoder()
bm25.fit(metadata['productDisplayName'])

# Dense encoder
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device=device)

# Create sparse & dense vectors
batch_size = 100
fashion_data_num = 1000

for i in tqdm(range(0, min(fashion_data_num, len(fashion)), batch_size)):
    i_end = min(i + batch_size, len(fashion))
    meta_batch = metadata.iloc[i:i_end]
    meta_dict = meta_batch.to_dict(orient="records")
    meta_texts = [" ".join(x) for x in meta_batch.drop(columns=['id', 'year'], errors='ignore').values.tolist()]
    img_batch = images[i:i_end]

    sparse_embeds = bm25.encode_documents(meta_texts)
    dense_embeds = model.encode(img_batch).tolist()
    ids = [str(x) for x in range(i, i_end)]

    upserts = []
    for _id, sparse, dense, meta in zip(ids, sparse_embeds, dense_embeds, meta_dict):
        upserts.append({
            'id': _id,
            'sparse_values': sparse,
            'values': dense,
            'metadata': meta
        })

    index.upsert(upserts)

# Show index stats
print(index.describe_index_stats())


# Query Function
def hybrid_scale(dense, sparse, alpha: float):
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")
    hsparse = {
        'indices': sparse['indices'],
        'values': [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse


def display_result(image_batch):
    figures = []
    for img in image_batch:
        b = BytesIO()
        img.save(b, format='png')
        figures.append(f'''
            <figure style="margin: 5px !important;">
              <img src="data:image/png;base64,{b64encode(b.getvalue()).decode('utf-8')}" style="width: 90px; height: 120px" >
            </figure>
        ''')
    return HTML(data=f'''
        <div style="display: flex; flex-flow: row wrap; text-align: center;">
        {''.join(figures)}
        </div>
    ''')


# Perform a query
query = "dark blue french connection jeans for men"
sparse = bm25.encode_queries(query)
dense = model.encode(query).tolist()

# Try alpha = 1 (Dense only)
hdense, hsparse = hybrid_scale(dense, sparse, alpha=1)
result = index.query(top_k=6, vector=hdense, sparse_vector=hsparse, include_metadata=True)
imgs = [images[int(r["id"])] for r in result["matches"]]

display_result(imgs)
for r in result["matches"]:
    print(r['metadata']['productDisplayName'])

# Try alpha = 0 (Sparse only)
hdense, hsparse = hybrid_scale(dense, sparse, alpha=0)
result = index.query(top_k=6, vector=hdense, sparse_vector=hsparse, include_metadata=True)
imgs = [images[int(r["id"])] for r in result["matches"]]

display_result(imgs)
for r in result["matches"]:
    print(r['metadata']['productDisplayName'])
