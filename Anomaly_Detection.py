import os
import time
from tqdm import tqdm
import torch
import pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, InputExample, models, losses
from torch import nn
from torch.utils.data import DataLoader
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env
load_dotenv()

# Get Pinecone API key directly
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in environment")

# Create consistent index name
INDEX_NAME = "dl-ai"

# Initialize Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY)

# Delete existing index (if any) and create new one
if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
    pinecone.delete_index(INDEX_NAME)

pinecone.create_index(
    name=INDEX_NAME,
    dimension=256,
    metric='cosine',
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
)

index = pinecone.Index(INDEX_NAME)

# Set paths for your files
base_path = "C:\\Users\\Acer\\Desktop\\AI_AGENT\\Pinecone\\lesson6"
training_path = os.path.join(base_path, "training.txt")
sample_log_path = os.path.join(base_path, "sample.log")

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Build SentenceTransformer model
word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=768)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(
    in_features=pooling_model.get_sentence_embedding_dimension(),
    out_features=256,
    activation_function=nn.Tanh()
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device=device)

# Load training data
train_examples = []
with open(training_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            a, b, label = line.split('^')
            train_examples.append(InputExample(texts=[a, b], label=float(label)))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# Train model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=16, warmup_steps=100)

# Load sample logs
samples = []
with open(sample_log_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            samples.append(line)

# Encode samples
embeddings = model.encode(samples)

# Prepare vectors for Pinecone
to_upsert = []
for i in tqdm(range(len(samples))):
    to_upsert.append({
        'id': str(i),
        'values': embeddings[i].tolist(),
        'metadata': {'log': samples[i]}
    })

index.upsert(to_upsert)

# Query for anomaly detection
good_log = samples[0]
print("Querying with good log line:")
print(good_log)

results = []
while len(results) == 0:
    time.sleep(2)
    response = index.query(
        vector=embeddings[0].tolist(),
        top_k=100,
        include_metadata=True
    )
    results = response['matches']
    print(".:. ", end="")

print("\nTop Similar Logs:")
for i in range(10):
    print(f"{round(results[i]['score'], 4)}\t{results[i]['metadata']['log']}")

# Show most anomalous log
print("\nMost Anomalous Log:")
print(f"{round(results[-1]['score'], 4)}\t{results[-1]['metadata']['log']}")
