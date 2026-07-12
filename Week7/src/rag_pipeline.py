"""
RAG Pipeline

Coordinates the complete Retrieval-Augmented Generation workflow:

1. Document Loading
2. Text Chunking
3. Embedding Generation
4. FAISS Vector Store Creation
5. Similarity Retrieval
6. Answer Generation
"""

import os
import time

from src.document_loader import load_documents
from src.chunking import chunk_documents
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.generator import Generator

import config


class RAGPipeline:
    def __init__(self):
        self.embedding_model = EmbeddingModel(config.EMBEDDING_MODEL_NAME)
        self.generator = Generator(config.LLM_BACKEND)
        self.store = None

        # Store system metrics
        self.metrics = {}

    # ==========================================================
    # BUILD VECTOR DATABASE
    # ==========================================================

    def ingest(self, data_dir: str = config.DATA_DIR):

        total_start = time.time()

        print("\n" + "=" * 65)
        print("DOCUMENT INGESTION")
        print("=" * 65)

        # ---------------- Load Documents ----------------

        load_start = time.time()

        documents = load_documents(data_dir)

        load_time = time.time() - load_start

        if len(documents) == 0:
            raise ValueError("No supported documents found.")

        total_characters = sum(len(text) for text in documents.values())

        print(f"Documents Loaded     : {len(documents)}")
        print(f"Characters           : {total_characters:,}")
        print(f"Loading Time         : {load_time:.2f} sec")

        # ---------------- Chunking ----------------

        chunk_start = time.time()

        chunks = chunk_documents(
            documents,
            config.CHUNK_SIZE,
            config.CHUNK_OVERLAP,
        )

        chunk_time = time.time() - chunk_start

        print("\nChunking Complete")
        print(f"Chunks Created       : {len(chunks)}")
        print(f"Chunk Size           : {config.CHUNK_SIZE}")
        print(f"Chunk Overlap        : {config.CHUNK_OVERLAP}")
        print(f"Chunking Time        : {chunk_time:.2f} sec")

        # ---------------- Embeddings ----------------

        embedding_start = time.time()

        texts = [c["text"] for c in chunks]

        embeddings = self.embedding_model.embed_texts(texts)

        embedding_time = time.time() - embedding_start

        embedding_dim = embeddings.shape[1]

        print("\nEmbedding Complete")
        print(f"Embedding Model      : {config.EMBEDDING_MODEL_NAME}")
        print(f"Embedding Dimension  : {embedding_dim}")
        print(f"Embedding Time       : {embedding_time:.2f} sec")

        # ---------------- Vector Store ----------------

        vector_start = time.time()

        self.store = VectorStore(dim=embedding_dim)

        self.store.add(embeddings, chunks)

        self.store.save(config.INDEX_PATH, config.META_PATH)

        vector_time = time.time() - vector_start

        print("\nVector Store")
        print(f"Database             : FAISS")
        print(f"Index Saved          : {config.INDEX_PATH}")
        print(f"Vector Time          : {vector_time:.2f} sec")

        total_time = time.time() - total_start

        print("\n" + "=" * 65)
        print("INGESTION COMPLETED SUCCESSFULLY")
        print("=" * 65)
        print(f"Total Time           : {total_time:.2f} sec")
        print("=" * 65)

        # Save metrics for Streamlit UI

        self.metrics = {
            "documents": len(documents),
            "characters": total_characters,
            "chunks": len(chunks),
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "embedding_dimension": embedding_dim,
            "vector_database": "FAISS",
            "ingestion_time": round(total_time, 2),
        }

    # ==========================================================
    # LOAD EXISTING VECTOR STORE
    # ==========================================================

    def load_index(self):

        if not os.path.exists(config.INDEX_PATH):
            raise FileNotFoundError(
                "Vector index not found. Run ingestion first."
            )

        self.store = VectorStore.load(
            config.INDEX_PATH,
            config.META_PATH,
        )

        print("✓ Vector index loaded successfully.")

    # ==========================================================
    # QUESTION ANSWERING
    # ==========================================================

    def query(
        self,
        question: str,
        top_k: int = config.TOP_K,
    ) -> dict:

        if self.store is None:
            self.load_index()

        print("\n" + "=" * 65)
        print("QUESTION ANSWERING")
        print("=" * 65)

        print(f"Question             : {question}")

        # ---------- Embed Question ----------

        query_start = time.time()

        query_embedding = self.embedding_model.embed_query(question)

        embedding_time = time.time() - query_start

        # ---------- Retrieval ----------

        retrieval_start = time.time()

        retrieved = self.store.search(
            query_embedding,
            top_k,
        )

        retrieval_time = time.time() - retrieval_start

        # ---------- Generation ----------

        generation_start = time.time()

        answer = self.generator.generate(
            question,
            retrieved,
        )

        generation_time = time.time() - generation_start

        total_time = (
            embedding_time +
            retrieval_time +
            generation_time
        )

        print(f"\nTop-K Retrieved      : {len(retrieved)}")
        print(f"Embedding Time       : {embedding_time:.2f} sec")
        print(f"Retrieval Time       : {retrieval_time:.2f} sec")
        print(f"Generation Time      : {generation_time:.2f} sec")
        print(f"Total Response Time  : {total_time:.2f} sec")

        print("=" * 65)

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved,
            "metrics": {
                "top_k": len(retrieved),
                "embedding_time": round(embedding_time, 3),
                "retrieval_time": round(retrieval_time, 3),
                "generation_time": round(generation_time, 3),
                "total_time": round(total_time, 3),
            },
        }