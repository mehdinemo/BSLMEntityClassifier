import json
import os
from typing import List

import datasets
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

hf_token = os.getenv("HF_TOKEN")


def query(texts: List[str], model_id):
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
    return response.json()


def get_most_similar_pairs(model_id, texts, sim_thresh=0.7, hf_hub=True):
    if hf_hub:
        embeddings = query(texts, model_id)
    else:
        model = SentenceTransformer(model_id)
        embeddings = model.encode(texts)
    
    cosine_sim_matrix = cosine_similarity(embeddings)
    most_similar_pairs = np.argwhere(cosine_sim_matrix > sim_thresh)

    similar_pairs = []
    for sim_pair in most_similar_pairs:
        if sim_pair[0] < sim_pair[1]:
            similar_pairs.append({
                "titles": [
                    texts[sim_pair[0]],
                    texts[sim_pair[1]]
                ],
                "similarity": cosine_sim_matrix[sim_pair[0], sim_pair[1]]
            })

    return similar_pairs


def add_normalized_title(products):
    products["normalized_title"] = normalize_texts(products['title'])
    return products


dataset = datasets.load_dataset('BaSalam/bslm-product-entity-cls-610k', split="train")
dataset = dataset.train_test_split(test_size=100, shuffle=True)

try:
    from TextTools import normalize_texts

    dataset["test"] = dataset["test"].map(add_normalized_title, batched=True, batch_size=100)
    texts = dataset['test'].to_dict().get('normalized_title')
except:
    texts = dataset['test'].to_dict().get('title')

model_ids = [
    'sentence-transformers/LaBSE',
    'sentence-transformers/distiluse-base-multilingual-cased-v2',
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
]

results = {}
for model_id in model_ids:
    results.update({model_id: get_most_similar_pairs(model_id, texts, hf_hub=True)})

with open('compare_embedding_models', 'w') as f:
    print(json.dump(results, f, ensure_ascii=False))

# Best model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
