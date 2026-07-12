"""
Stage 4: Vector Database

Wrapper around FAISS for:
- Storing embeddings
- Similarity search
- Save / Load
- Retrieval statistics
"""

import os
import pickle
import numpy as np
import faiss


class VectorStore:

    def __init__(self, dim: int):

        self.dimension = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

    # =====================================================
    # ADD DOCUMENTS
    # =====================================================

    def add(
        self,
        embeddings: np.ndarray,
        metadata: list,
    ):

        if len(embeddings) != len(metadata):
            raise ValueError(
                "Embeddings and metadata length mismatch."
            )

        self.index.add(embeddings)

        self.metadata.extend(metadata)

        print(f"✓ Added {len(metadata)} vectors to FAISS index.")

    # =====================================================
    # SEARCH
    # =====================================================

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list:

        if self.index.ntotal == 0:
            raise RuntimeError(
                "Vector database is empty."
            )

        query_embedding = np.expand_dims(
            query_embedding,
            axis=0,
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        print("\nRetrieved Chunks")
        print("-" * 60)

        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):

            if idx == -1:
                continue

            item = dict(self.metadata[idx])

            item["score"] = float(score)

            item["rank"] = rank

            results.append(item)

            print(
                f"{rank}. "
                f"{item['source']} | "
                f"Chunk {item['chunk_id']} | "
                f"Score {score:.4f}"
            )

        print("-" * 60)

        return results

    # =====================================================
    # SAVE
    # =====================================================

    def save(
        self,
        index_path: str,
        meta_path: str,
    ):

        os.makedirs(
            os.path.dirname(index_path),
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            index_path,
        )

        with open(
            meta_path,
            "wb",
        ) as f:

            pickle.dump(
                self.metadata,
                f,
            )

        print("✓ Vector database saved.")

    # =====================================================
    # LOAD
    # =====================================================

    @classmethod
    def load(
        cls,
        index_path: str,
        meta_path: str,
    ):

        if not os.path.exists(index_path):
            raise FileNotFoundError(index_path)

        if not os.path.exists(meta_path):
            raise FileNotFoundError(meta_path)

        index = faiss.read_index(index_path)

        store = cls.__new__(cls)

        store.index = index

        store.dimension = index.d

        with open(
            meta_path,
            "rb",
        ) as f:

            store.metadata = pickle.load(f)

        print(
            f"✓ Loaded FAISS index with "
            f"{store.index.ntotal} vectors."
        )

        return store

    # =====================================================
    # SYSTEM INFO
    # =====================================================

    @property
    def stats(self):

        return {
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal,
            "metadata_entries": len(self.metadata),
        }