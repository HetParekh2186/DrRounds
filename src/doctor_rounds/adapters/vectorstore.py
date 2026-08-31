"""Vector-store adapters.

`VectorStore` is a `Protocol`: anything with `add`/`retrieve` methods
works, so evaluating a RAG pipeline never requires migrating it onto this
project's own storage layer. `ChromaVectorStore` exists so there's a
zero-external-service way to try Doctor Rounds — Chroma runs embedded, in
memory or on local disk, no server to stand up.
"""

from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable

from doctor_rounds.core.types import Chunk, RetrievedChunk


@runtime_checkable
class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> None: ...
    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]: ...


class ChromaVectorStore:
    """A `VectorStore` backed by a local Chroma collection.

    Requires `pip install doctor-rounds[chroma]`. Runs fully offline once
    the embedding model is downloaded — no external vector-DB service.
    """

    def __init__(
        self,
        collection_name: str = "doctor_rounds",
        embedding_model: str = "all-MiniLM-L6-v2",
        # chromadb's own EmbeddingFunction type is exported inconsistently
        # between chromadb.api.types and chromadb.utils.embedding_functions
        # (mypy sees them as distinct, incompatible generics) — Any avoids
        # fighting that rather than papering over it with a cast.
        embedding_function: Any | None = None,
        persist_directory: str | None = None,
    ) -> None:
        """
        Args:
            collection_name: Chroma collection to create/reuse.
            embedding_model: sentence-transformers model name, used unless
                `embedding_function` is given explicitly.
            embedding_function: overrides `embedding_model` entirely — for
                bringing your own embedder (e.g. an API-based one).
            persist_directory: if given, the index survives across
                processes; otherwise it's in-memory only (fine for a demo
                or a single eval run).
        """
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self._client = (
            chromadb.PersistentClient(path=persist_directory)
            if persist_directory
            else chromadb.EphemeralClient()
        )
        ef = embedding_function or SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            # See the embedding_function param's docstring note: chromadb's
            # own stubs disagree with themselves on this type across
            # modules, so this cast reflects a library inconsistency, not
            # an unsound assumption of ours.
            embedding_function=cast(Any, ef),
        )
        # Chroma's own metadata store only holds flat scalar values; the
        # full Chunk (arbitrary metadata dict included) is kept here so
        # retrieve() can return it faithfully.
        self._chunks_by_id: dict[str, Chunk] = {}

    def add(self, chunks: list[Chunk], batch_size: int = 512) -> None:
        """Indexes `chunks`. Batched because embedding (and Chroma's own
        insert path) is where the real cost is for a large corpus — one
        giant call risks timing out or exhausting memory on tens of
        thousands of passages."""
        # range() over a non-empty `chunks` never produces a start index
        # past the end, so every slice below is non-empty; an empty
        # `chunks` makes the range itself empty. No empty-batch case to
        # guard.
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[{"source": c.source or ""} for c in batch],
            )
            for c in batch:
                self._chunks_by_id[c.id] = c

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        results = self._collection.query(query_texts=[query], n_results=k)
        ids: list[str] = results["ids"][0]
        documents_field = results.get("documents")
        # Chroma's query result types (list[str] for a populated field vs.
        # a bare `None` when the field wasn't requested) are narrower than
        # what we assign them into here (`| None` per element, to allow the
        # fallback) — lists are invariant in their element type, so mypy
        # can't see that widening a read-only list this way is safe; cast
        # reflects that, not an unsound assumption.
        documents = cast(
            "list[str | None]", documents_field[0] if documents_field else [None] * len(ids)
        )
        distances_field = results.get("distances")
        distances = cast(
            "list[float | None]", distances_field[0] if distances_field else [None] * len(ids)
        )

        retrieved: list[RetrievedChunk] = []
        for rank, chunk_id in enumerate(ids):
            chunk = self._chunks_by_id.get(chunk_id)
            if chunk is None:
                # Store was populated in a different process (persisted
                # index) — reconstruct a minimal Chunk from what Chroma
                # itself has, rather than failing.
                chunk = Chunk(id=chunk_id, text=documents[rank] or "")
            distance = distances[rank]
            # Chroma returns a distance (lower = more similar); score is
            # reported as similarity (higher = better) to match the sign
            # convention the rest of this project's metrics assume.
            score = 1.0 - distance if distance is not None else None
            retrieved.append(RetrievedChunk(chunk=chunk, rank=rank, score=score))
        return retrieved
