import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Literal, List, Optional

import datasets
import numpy as np
import torch
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from TextTools import normalize_texts
from constants import EMBEDDING_MODEL_NAME, CHROMA_PERSIST_DIRECTORY
from utils import logger_function

logger = logger_function(name="investigate_approaches")


def similarity(embeddings, texts, threshold=0.8):
    cosine_sim_matrix = cosine_similarity(embeddings)
    most_similar_pairs = np.argwhere(cosine_sim_matrix > threshold)  # Threshold for similarity

    # Apply DBSCAN clustering on the embeddings
    clustering_model = DBSCAN(metric='cosine', eps=0.5, min_samples=2)
    clusters = clustering_model.fit_predict(embeddings)

    # Print the cluster assignments
    for i, cluster in enumerate(clusters):
        print(f"Product {i} belongs to cluster {cluster}")

    # Extract most common entities within each cluster
    for cluster_id in set(clusters):
        cluster_items = [texts[i] for i in range(len(clusters)) if clusters[i] == cluster_id]
        print(f"Cluster {cluster_id}: {Counter(' '.join(cluster_items).split()).most_common(5)}")


class Embeddings:
    logger = logger_function('EmbeddingsVS')

    def __init__(
            self,
            model_id: str,
            cache_folder: Optional[str] = None,
            method: Literal["local", "hf_hub"] = "local",
            device: Optional[str] = None,
            normalize_embeddings: bool = False
    ):
        self.model = self._get_embedding_model(model_id, cache_folder, method, device, normalize_embeddings)

    def _get_embedding_model(self, model_id, cache_folder, method, device, normalize_embeddings):
        self.logger.info(f"Initialization embedding model...")
        if method == "hf_hub":
            hf_token = os.getenv("HF_TOKEN")
            if hf_token is None:
                raise ValueError("When using HuggingFace Hub, you need to set the 'HF_TOKEN' environment variable")

            # api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
            # headers = {"Authorization": f"Bearer {hf_token}"}

            return HuggingFaceInferenceAPIEmbeddings(api_key=hf_token, model_name=model_id)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("No GPU found, using cpu.")
            device = "cpu"

        self.logger.info(f"Using {device}")

        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': normalize_embeddings}

        return HuggingFaceEmbeddings(
            model_name=model_id,
            cache_folder=cache_folder,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    @staticmethod
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def creating_local_vectorstore(
            self,
            texts: List[str],
            persist_directory: str,
            reset_cache: bool = False,
            batch_size: Optional[int] = None
    ) -> Chroma:
        """
        Create a Chroma vectorstore and save it on disk.
        If `persist_directory` is specified, the collection will be saved in it.
        If `reset_cache` is set to `True`, all files in the `persist_directory` will be deleted before building the new Chroma vectorstore.

        :param batch_size:
        :param texts:
        :param persist_directory: Directory to persist the Chroma vectorstore.
        :param reset_cache: `bool` default `False`. If True, delete all files in the persist_directory before building the new Chroma vectorstore.
        :return: Chroma
        """

        if not reset_cache and len(os.listdir(persist_directory)) > 0:
            self.logger.info(f'Already exists in: {persist_directory}')
            return

        if reset_cache:
            self.logger.warning('Delete all files in the persist_directory')
            for filename in os.listdir(persist_directory):
                file_path = os.path.join(persist_directory, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

        self.logger.info('Creating vectorstore...')
        if batch_size:
            vs = Chroma.from_texts(
                texts=texts[:batch_size],
                embedding=self.model,
                persist_directory=persist_directory
            )
            for batch_texts in tqdm(self.batch(texts[batch_size:], batch_size), total=len(texts) / batch_size):
                vs.add_texts(batch_texts)
        else:
            vs = Chroma.from_texts(
                texts=texts,
                embedding=self.model,
                persist_directory=persist_directory
            )
        self.logger.info(f'Chroma vectorstore created in: {persist_directory}')

        return vs


def add_normalized_title(titles, entities):
    return {
        "normalized_title": normalize_texts(titles),
        "normalized_entity": normalize_texts(entities)
    }


def normalize_titles_and_entities():
    dataset = datasets.load_dataset('BaSalam/bslm-product-entity-cls-610k', split="train")
    dataset = dataset.map(
        add_normalized_title,
        input_columns=['title', 'entity'],
        remove_columns=['title', 'description', 'entity'],
        batched=True,
        batch_size=1000
    )
    dataset.save_to_disk(dataset_path='dataset/normalized')


def embedd_titles_and_entities():
    dataset = datasets.load_from_disk(dataset_path='dataset/normalized')
    dataset = dataset.to_dict()

    embeddings = Embeddings(
        model_id=EMBEDDING_MODEL_NAME,
        method="local"
    )

    titles_persist_directory = os.path.join(CHROMA_PERSIST_DIRECTORY, 'titles')
    Path(titles_persist_directory).mkdir(parents=True, exist_ok=True)
    embeddings.creating_local_vectorstore(
        texts=dataset.get('normalized_title'),
        persist_directory=titles_persist_directory,
        reset_cache=True,
        batch_size=1000
    )

    entities_persist_directory = os.path.join(CHROMA_PERSIST_DIRECTORY, 'entities')
    Path(entities_persist_directory).mkdir(parents=True, exist_ok=True)
    embeddings.creating_local_vectorstore(
        texts=dataset.get('normalized_entity'),
        persist_directory=entities_persist_directory,
        reset_cache=True,
        batch_size=1000
    )


def main():
    embeddings = Embeddings(
        model_id=EMBEDDING_MODEL_NAME,
        method="local"
    )

    titles_persist_directory = os.path.join(CHROMA_PERSIST_DIRECTORY, 'titles')
    x_vectorstore = Chroma(
        persist_directory=titles_persist_directory,
        embedding_function=embeddings.model
    )
    entities_persist_directory = os.path.join(CHROMA_PERSIST_DIRECTORY, 'titles')
    y_vectorstore = Chroma(
        persist_directory=entities_persist_directory,
        embedding_function=embeddings.model
    )

    x_embeddings = x_vectorstore.get(limit=10000, include=['embeddings']).get('embeddings')
    y_embeddings = y_vectorstore.get(limit=10000, include=['embeddings']).get('embeddings')

    X_train, X_test, y_train, y_test = train_test_split(
        x_embeddings,
        y_embeddings,
        test_size=0.2,
        random_state=42
    )

    # Train a classifier (Logistic Regression in this case)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

    # Classification report for more detailed metrics
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    # embedd_titles_and_entities()
    main()
