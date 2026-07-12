"""
Stage 3: Embedding Creation

Converts text into dense vector representations using
Sentence Transformers for semantic similarity search.
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self, model_name: str):

        self.model_name = model_name

        print("\nLoading Embedding Model")
        print("-" * 60)

        start = time.time()

        self.model = SentenceTransformer(model_name)

        load_time = time.time() - start

        self.dimension = self.model.get_sentence_embedding_dimension()

        print(f"Model                : {model_name}")
        print(f"Embedding Dimension  : {self.dimension}")
        print(f"Load Time            : {load_time:.2f} sec")
        print("-" * 60)

    # =====================================================
    # EMBED MULTIPLE TEXTS
    # =====================================================

    def embed_texts(
        self,
        texts: list,
    ) -> np.ndarray:

        if len(texts) == 0:
            raise ValueError("No text provided for embedding.")

        start = time.time()

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        elapsed = time.time() - start

        print(
            f"Generated {len(texts)} embeddings "
            f"in {elapsed:.2f} sec"
        )

        return embeddings.astype(np.float32)

    # =====================================================
    # EMBED QUERY
    # =====================================================

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        return self.embed_texts([query])[0]

    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    @property
    def info(self):

        return {
            "model": self.model_name,
            "dimension": self.dimension,
        }