

import os
from pinecone import Pinecone

# Get your API key from https://app.pinecone.io and replace "PINECONE_API_KEY"
api_key = os.environ.get("PINECONE_API_KEY") or "pcsk_tyjM4_52iX3ecBD3agBMMmwXcy6ytx6vVNJ7aQECYx1hB9dj3S81oLzJAsn4hBytMSuoK"

# Instantiate Pinecone client
pc = Pinecone(api_key=api_key)

from pinecone import ServerlessSpec, CloudProvider, AwsRegion, Metric

index_name = "hello-pinecone"

# Delete existing index if any
if pc.has_index(name=index_name):
    pc.delete_index(name=index_name)

# Create index
pc.create_index(
    name=index_name,
    metric=Metric.COSINE,
    dimension=3,
    spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
)
description = pc.describe_index(name=index_name)
print(description)
# Instantiate index client
index = pc.Index(host=description.host)

import random
import pandas as pd

def create_simulated_data_in_df(num_vectors):
    df = pd.DataFrame({
        "id": [f"id-{i}" for i in range(num_vectors)],
        "vector": [
            [random.random() for _ in range(description.dimension)]
            for _ in range(num_vectors)
        ],
    })
    return df

df = create_simulated_data_in_df(10)

# View some vectors
print(df.head())

# Upsert into index
index.upsert(vectors=zip(df.id, df.vector))
import time

def is_fresh(index):
    stats = index.describe_index_stats()
    vector_count = stats.total_vector_count
    print(f"Vector count: {vector_count}")
    return vector_count > 0

while not is_fresh(index):
    time.sleep(5)
print(index.describe_index_stats())
# Sample query vector
query_embedding = [2.0, 2.0, 2.0]

# Query top 5 matches
results = index.query(vector=query_embedding, top_k=5, include_values=True)

# Show results
from pprint import pprint
pprint(results)
