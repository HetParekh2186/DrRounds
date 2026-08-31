"""Tests for doctor_rounds.adapters.vectorstore.ChromaVectorStore.

Most tests use a small deterministic fake embedder — hashing each
document's word set into a fixed-size vector — so they run fast and
offline, exercising the adapter's own logic (batching, ID bookkeeping,
score-sign convention, empty-input handling) without depending on a real
embedding model producing good semantics. A single `integration`-marked
test at the bottom uses the real sentence-transformers model to confirm
actual semantic retrieval quality end to end.
"""

import hashlib
import uuid

import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from doctor_rounds.adapters.vectorstore import ChromaVectorStore
from doctor_rounds.core.types import Chunk


class FakeEmbeddingFunction(EmbeddingFunction):
    """Deterministic, dependency-free stand-in for a real embedder.
    Same text always maps to the same vector, and Chroma only needs a
    callable — no semantic quality guarantees, which is fine since these
    tests check the adapter's plumbing, not retrieval quality.

    Subclasses chromadb's real `EmbeddingFunction` (rather than just
    duck-typing `__call__`) so default implementations of everything else
    Chroma's internals expect (`embed_query`, etc.) come for free.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(text) for text in input]  # type: ignore[misc]

    @staticmethod
    def _embed(text: str, dims: int = 16) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in digest[:dims]]

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "FakeEmbeddingFunction":
        return FakeEmbeddingFunction()

    @staticmethod
    def name() -> str:
        return "fake-embedding-function-for-tests"


@pytest.fixture
def store() -> ChromaVectorStore:
    return ChromaVectorStore(
        collection_name=f"test-{uuid.uuid4().hex}",  # unique per test, avoids cross-test collisions
        embedding_function=FakeEmbeddingFunction(),
    )


SAMPLE_CHUNKS = [
    Chunk(id="c1", text="Metformin is first-line therapy for type 2 diabetes.", source="doc-a"),
    Chunk(id="c2", text="Aspirin inhibits platelet aggregation.", source="doc-b"),
    Chunk(id="c3", text="Beta blockers reduce heart rate and blood pressure.", source="doc-c"),
]


class TestAdd:
    def test_indexes_all_chunks(self, store: ChromaVectorStore):
        store.add(SAMPLE_CHUNKS)
        assert len(store.retrieve("anything", k=100)) == 3

    def test_empty_list_is_a_no_op(self, store: ChromaVectorStore):
        store.add([])
        assert store.retrieve("anything", k=100) == []

    def test_batches_large_inputs(self, store: ChromaVectorStore):
        many = [Chunk(id=f"c{i}", text=f"chunk number {i}") for i in range(10)]
        store.add(many, batch_size=3)
        assert len(store.retrieve("anything", k=100)) == 10


class TestRetrieve:
    def test_returns_k_results(self, store: ChromaVectorStore):
        store.add(SAMPLE_CHUNKS)
        results = store.retrieve("diabetes treatment", k=2)
        assert len(results) == 2

    def test_ranks_are_sequential_from_zero(self, store: ChromaVectorStore):
        store.add(SAMPLE_CHUNKS)
        results = store.retrieve("diabetes treatment", k=3)
        assert [r.rank for r in results] == [0, 1, 2]

    def test_retrieved_chunk_matches_indexed_chunk(self, store: ChromaVectorStore):
        store.add(SAMPLE_CHUNKS)
        results = store.retrieve("aspirin platelets", k=3)
        ids = {r.chunk.id for r in results}
        assert ids == {"c1", "c2", "c3"}
        by_id = {r.chunk.id: r.chunk for r in results}
        assert by_id["c2"].text == "Aspirin inhibits platelet aggregation."
        assert by_id["c2"].source == "doc-b"

    def test_score_is_higher_for_closer_match(self, store: ChromaVectorStore):
        # querying with the exact text of c1 should score c1 highest,
        # since distance-to-self is the minimum possible distance
        store.add(SAMPLE_CHUNKS)
        results = store.retrieve(SAMPLE_CHUNKS[0].text, k=3)
        by_id = {r.chunk.id: r.score for r in results}
        assert by_id["c1"] == max(by_id.values())

    def test_k_larger_than_corpus_returns_all(self, store: ChromaVectorStore):
        store.add(SAMPLE_CHUNKS)
        results = store.retrieve("anything", k=100)
        assert len(results) == 3

    def test_reconstructs_chunk_from_persisted_store_in_a_fresh_process(self, tmp_path):
        # Simulates the real scenario the fallback in retrieve() exists
        # for: index once, then query from a *different* ChromaVectorStore
        # instance (standing in for a different process) whose in-memory
        # _chunks_by_id cache was never populated by add().
        persist_dir = str(tmp_path / "chroma")
        writer = ChromaVectorStore(
            collection_name="persisted",
            embedding_function=FakeEmbeddingFunction(),
            persist_directory=persist_dir,
        )
        writer.add(SAMPLE_CHUNKS)

        reader = ChromaVectorStore(
            collection_name="persisted",
            embedding_function=FakeEmbeddingFunction(),
            persist_directory=persist_dir,
        )
        assert reader._chunks_by_id == {}  # fresh instance, nothing added via this object

        results = reader.retrieve(SAMPLE_CHUNKS[0].text, k=1)
        assert results[0].chunk.id == "c1"
        assert results[0].chunk.text == SAMPLE_CHUNKS[0].text
        # source isn't part of Chroma's document text, so the reconstructed
        # fallback Chunk legitimately won't carry it — documents the
        # tradeoff rather than hiding it
        assert results[0].chunk.source is None


@pytest.mark.integration
class TestRealSemanticRetrieval:
    """Uses the real sentence-transformers model — downloads ~90MB on
    first run, so kept out of the default fast test suite."""

    def test_semantically_related_query_outranks_unrelated_chunk(self):
        store = ChromaVectorStore(collection_name="test-real-semantics")
        store.add(
            [
                Chunk(id="relevant", text="Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID)."),
                Chunk(id="unrelated", text="The Eiffel Tower was completed in 1889 in Paris."),
            ]
        )
        results = store.retrieve("What kind of drug is ibuprofen?", k=2)
        assert results[0].chunk.id == "relevant"
