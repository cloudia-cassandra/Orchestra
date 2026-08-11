"""Lazy singleton Chroma client and collection, configured from CHROMA_HOST/CHROMA_PORT."""

import os

import chromadb

COLLECTION_NAME = "orchestra_long_term_memory"

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        host = os.environ.get("CHROMA_HOST", "localhost")
        port = int(os.environ.get("CHROMA_PORT", 8000))
        _client = chromadb.HttpClient(host=host, port=port)
    return _client


def get_memory_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        _collection = get_chroma_client().get_or_create_collection(COLLECTION_NAME)
    return _collection