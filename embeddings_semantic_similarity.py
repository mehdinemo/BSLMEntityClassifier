import json
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
from sklearn.cluster import spectral_clustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from constants import EMBEDDING_MODEL_NAME, CHROMA_PERSIST_DIRECTORY, PATH_OR_NAME_OF_THE_DATASET
from text_tools import remove_stop_words
from utils import logger_function

logger = logger_function(name="embeddings_semantic_similarity")


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
            self.logger.info(f"Already exists in '{persist_directory}'")
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=self.model
            )

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


def add_normalized_title(titles, entities, normalize_fn):
    return {
        "title": normalize_fn(titles),
        "entity": normalize_fn(entities)
    }


def normalize_titles_and_entities():
    try:
        from TextTools import normalize_texts
    except:
        logger.exception('Could not import TextTools')
        return

    dataset = datasets.load_dataset(PATH_OR_NAME_OF_THE_DATASET, split="train")
    dataset = dataset.map(
        add_normalized_title,
        input_columns=['title', 'entity'],
        remove_columns=['title', 'description', 'entity'],
        batched=True,
        batch_size=1000,
        fn_kwargs={"normalize_fn": normalize_texts}
    )

    dataset.save_to_disk(dataset_path='dataset/normalized')


def embedd_titles_and_entities():
    try:
        dataset = datasets.load_from_disk(dataset_path='dataset/normalized')
    except:
        logger.warning('Could not load normalized dataset. using original dataset.')
        dataset = datasets.load_dataset(PATH_OR_NAME_OF_THE_DATASET, split="train")

    dataset = dataset.to_dict()

    embeddings = Embeddings(model_id=EMBEDDING_MODEL_NAME, method="local")

    titles_persist_directory = os.path.join(CHROMA_PERSIST_DIRECTORY, 'titles')
    Path(titles_persist_directory).mkdir(parents=True, exist_ok=True)
    embeddings.creating_local_vectorstore(
        texts=dataset.get('title'),
        persist_directory=titles_persist_directory,
        reset_cache=False,
        batch_size=1000
    )

    entities_persist_directory = os.path.join(CHROMA_PERSIST_DIRECTORY, 'entities')
    Path(entities_persist_directory).mkdir(parents=True, exist_ok=True)
    embeddings.creating_local_vectorstore(
        texts=dataset.get('entity'),
        persist_directory=entities_persist_directory,
        reset_cache=False,
        batch_size=1000
    )


def classification():
    """
    *********** Not completed ***********
    """

    embeddings = Embeddings(model_id=EMBEDDING_MODEL_NAME, method="local")

    titles_persist_directory = os.path.join(CHROMA_PERSIST_DIRECTORY, 'titles')
    x_vectorstore = Chroma(
        persist_directory=titles_persist_directory,
        embedding_function=embeddings.model
    )

    entities_persist_directory = os.path.join(CHROMA_PERSIST_DIRECTORY, 'entities')
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


def clustering(n_samples=10000):
    """
    Cluster embeddings using spectral clustering.
    :param n_samples: Number of first samples to use for clustering
    :return: Array of clusters
    """

    embeddings = Embeddings(model_id=EMBEDDING_MODEL_NAME, method="local")

    titles_persist_directory = os.path.join(CHROMA_PERSIST_DIRECTORY, 'titles')
    titles_vectorstore = Chroma(
        persist_directory=titles_persist_directory,
        embedding_function=embeddings.model
    )

    titles_vectors = titles_vectorstore.get(limit=n_samples, include=['embeddings']).get('embeddings')

    cosine_sim_matrix = cosine_similarity(titles_vectors)
    labels = spectral_clustering(cosine_sim_matrix, n_clusters=int(n_samples / 10), eigen_solver="arpack")

    return labels


def entity_extraction(n_samples=10000):
    try:
        dataset = datasets.load_from_disk(dataset_path='dataset/normalized')
    except:
        logger.warning('Could not load normalized dataset. Using original dataset.')
        dataset = datasets.load_dataset(PATH_OR_NAME_OF_THE_DATASET, split="train")

    dataset = dataset.to_dict()
    texts = dataset.get('title')
    texts = texts[:n_samples]

    labels_path = 'spectral_labels.npy'
    if os.path.isfile(labels_path):
        logger.info("Loading labels from disk.")
        labels = np.load(labels_path)
    else:
        logger.info("Starting Spectral Clustering. This take a while...")
        labels = clustering(n_samples)
        np.save(labels_path, labels)
        logger.info("Labels saved to disk.")

    # Extract most common entities within each cluster
    clusters_entities = []
    for cluster_id in set(labels):
        cluster_items = [remove_stop_words(texts[i]) for i in range(len(labels)) if labels[i] == cluster_id]
        common_elements = Counter(' '.join(cluster_items).split()).most_common(4)
        clusters_entities.append([element[0] for element in common_elements])

    with open('spectral_extracted_entities.json', 'w') as f:
        json.dump(clusters_entities, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # normalize_titles_and_entities()
    embedd_titles_and_entities()
    entity_extraction()
