import warnings
warnings.filterwarnings('ignore')

import os
from dotenv import load_dotenv
from deepface import DeepFace
from pinecone import Pinecone, ServerlessSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import contextlib

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Constants
DATASET_PATH = r"C:\Users\Acer\Desktop\AI_AGENT\Pinecone\family_photos\family"
VECTOR_FILE = "vectors.vec"
MODEL = "Facenet"
INDEX_NAME = "facial-similarity-index"

# Initialize Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY)

def show_img(path):
    img = plt.imread(path)
    plt.figure(figsize=(4, 3))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def generate_vectors():
    with contextlib.suppress(FileNotFoundError):
        os.remove(VECTOR_FILE)
    with open(VECTOR_FILE, "w") as f:
        for person in ["mom", "dad", "child"]:
            files = glob.glob(os.path.join(DATASET_PATH, person, "*"))
            for file in tqdm(files, desc=f"Processing {person}"):
                try:
                    embedding = DeepFace.represent(img_path=file, model_name=MODEL, enforce_detection=False)[0]['embedding']
                    f.write(f'{person}:{os.path.basename(file)}:{embedding}\n')
                except Exception as e:
                    print(f"Error processing {file}: {e}")

def gen_tsne_df(person, perplexity):
    vectors = []
    with open(VECTOR_FILE, 'r') as f:
        for line in f:
            p, _, v = line.strip().split(':')
            if p == person:
                vectors.append(eval(v))
    pca = PCA(n_components=8)
    tsne = TSNE(2, perplexity=perplexity, random_state=0, n_iter=1000, learning_rate=75)
    pca_transform = pca.fit_transform(vectors)
    embeddings2d = tsne.fit_transform(pca_transform)
    return pd.DataFrame({'x': embeddings2d[:, 0], 'y': embeddings2d[:, 1]})

def plot_tsne(perplexity, model):
    (_, ax) = plt.subplots(figsize=(8, 5))
    plt.grid(color='#EAEAEB', linewidth=0.5)
    ax.spines['top'].set_color(None)
    ax.spines['right'].set_color(None)
    ax.spines['left'].set_color('#2B2F30')
    ax.spines['bottom'].set_color('#2B2F30')
    colormap = {'dad': '#ee8933', 'child': '#4fad5b', 'mom': '#4c93db'}

    for person in colormap:
        embeddingsdf = gen_tsne_df(person, perplexity)
        ax.scatter(embeddingsdf.x, embeddingsdf.y, alpha=.5, label=person, color=colormap[person])

    plt.title(f'Scatter plot of faces using {model}', fontsize=16, fontweight='bold', pad=20)
    plt.suptitle(f't-SNE [perplexity={perplexity}]', y=0.92, fontsize=13)
    plt.legend(loc='best', frameon=True)
    plt.show()

def store_vectors():
    if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
        pinecone.delete_index(INDEX_NAME)
    pinecone.create_index(name=INDEX_NAME, dimension=128, metric='cosine',
                          spec=ServerlessSpec(cloud='aws', region='us-east-1'))
    index = pinecone.Index(INDEX_NAME)
    with open(VECTOR_FILE, "r") as f:
        for line in tqdm(f, desc="Uploading to Pinecone"):
            person, file, vec = line.strip().split(':')
            index.upsert([(f'{person}-{file}', eval(vec), {"person": person, "file": file})])

def test(vec_groups, parent, child):
    index = pinecone.Index(INDEX_NAME)
    parent_vecs = vec_groups[parent]
    K = 10
    SAMPLE_SIZE = 10
    total = 0
    for i in tqdm(range(SAMPLE_SIZE), desc=f"Querying {parent}"):
        query_response = index.query(
            top_k=K,
            vector=parent_vecs[i],
            filter={"person": {"$eq": child}}
        )
        for row in query_response["matches"]:
            total += row["score"]
    print(f"{parent} AVG Similarity to {child}: {total / (SAMPLE_SIZE * K)}")

def compute_scores():
    vec_groups = {"dad": [], "mom": [], "child": []}
    with open(VECTOR_FILE, "r") as f:
        for line in f:
            person, file, vec = line.strip().split(':')
            vec_groups[person].append(eval(vec))
    test(vec_groups, "dad", "child")
    test(vec_groups, "mom", "child")

def find_closest():
    index = pinecone.Index(INDEX_NAME)
    child_base = os.path.join(DATASET_PATH, "child", "P06310_face1.jpg")
    show_img(child_base)
    embedding = DeepFace.represent(img_path=child_base, model_name=MODEL)[0]['embedding']
    query_response = index.query(
        top_k=3,
        vector=embedding,
        filter={"person": {"$eq": "dad"}},
        include_metadata=True
    )
    print("Top Matches:", query_response)
    photo = query_response['matches'][0]['metadata']['file']
    show_img(os.path.join(DATASET_PATH, "dad", photo))

if __name__ == "__main__":
    generate_vectors()
    plot_tsne(44, MODEL)
    store_vectors()
    compute_scores()
    find_closest()
