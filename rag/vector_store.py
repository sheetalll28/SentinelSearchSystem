"""
Thin wrapper around ChromaDB so the rest of the app never touches the
vector-DB client directly. Swapping ChromaDB for Pinecone later only
requires rewriting this one file.
"""
from __future__ import annotations
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions

from config import CONFIG


class VectorStore:
    def __init__(self, collection_name: str = "sentinel_search"):
        self.client = chromadb.PersistentClient(path=CONFIG.vector_db_path)
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=CONFIG.embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedder
        )

    def add_chunks(self, chunks: List[str], metadatas: List[Dict], ids: List[str]):
        if not chunks:
            return
        # Chroma upsert in batches to avoid payload limits on large corpora
        batch = 128
        for i in range(0, len(chunks), batch):
            self.collection.upsert(
                documents=chunks[i:i + batch],
                metadatas=metadatas[i:i + batch],
                ids=ids[i:i + batch],
            )

    def query(self, question: str, top_k: int = 5) -> List[Dict]:
        if self.collection.count() == 0:
            return []
        res = self.collection.query(query_texts=[question], n_results=min(top_k, self.collection.count()))
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({"text": doc, "source": meta.get("source", "unknown"), "score": 1 - dist})
        return out

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name="sentinel_search", embedding_function=self.embedder
        )
