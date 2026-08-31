"""Tests for doctor_rounds.metrics.retrieval.

Each metric gets: a perfect case, a total-miss case, a partial case with a
hand-checked expected value, and at least one edge case (empty inputs, k
larger than the list, etc.) — the kind of boundary that silently breaks a
metrics library and then quietly corrupts every downstream eval report.
"""

import math

import pytest

from doctor_rounds.metrics.retrieval import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0

    def test_zero_recall(self):
        assert recall_at_k(["x", "y", "z"], {"a", "b"}, k=3) == 0.0

    def test_partial_recall(self):
        # only "a" of {"a", "b"} appears -> 1/2
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5

    def test_relevant_chunk_outside_k_does_not_count(self):
        # "b" is relevant but retrieved at rank 4, k=2 -> not counted
        assert recall_at_k(["a", "x", "y", "b"], {"a", "b"}, k=2) == 0.5

    def test_no_relevant_ids_returns_zero_not_nan(self):
        assert recall_at_k(["a", "b"], set(), k=3) == 0.0

    def test_k_larger_than_retrieved_list(self):
        assert recall_at_k(["a"], {"a"}, k=100) == 1.0

    def test_empty_retrieved_list(self):
        assert recall_at_k([], {"a"}, k=5) == 0.0


class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert precision_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_zero_precision(self):
        assert precision_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_partial_precision(self):
        # top 4: a (hit), x, y, b (hit) -> 2/4
        assert precision_at_k(["a", "x", "y", "b"], {"a", "b"}, k=4) == 0.5

    def test_k_zero_returns_zero(self):
        assert precision_at_k(["a", "b"], {"a"}, k=0) == 0.0

    def test_k_larger_than_list_divides_by_actual_length(self):
        # only 1 item retrieved even though k=10; precision is out of 1, not 10
        assert precision_at_k(["a"], {"a"}, k=10) == 1.0

    def test_empty_retrieved_list(self):
        assert precision_at_k([], {"a"}, k=5) == 0.0


class TestReciprocalRank:
    def test_hit_at_rank_one(self):
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_hit_at_rank_three(self):
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_hit_returns_zero(self):
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_first_relevant_hit_wins_when_multiple_present(self):
        assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5


class TestMeanReciprocalRank:
    def test_averages_correctly(self):
        assert mean_reciprocal_rank([1.0, 0.5, 0.0]) == pytest.approx(0.5)

    def test_empty_list_returns_zero(self):
        assert mean_reciprocal_rank([]) == 0.0


class TestNdcgAtK:
    def test_perfect_ranking_scores_one(self):
        # relevant chunks occupy the top positions in the ideal order too
        assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == pytest.approx(1.0)

    def test_zero_when_nothing_relevant_found(self):
        assert ndcg_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0

    def test_worse_rank_scores_lower_than_better_rank(self):
        # same recall, but "a" buried later should score strictly lower
        early = ndcg_at_k(["a", "x", "y"], {"a"}, k=3)
        late = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)
        assert early > late

    def test_hand_checked_value(self):
        # relevant={"a","b"}; retrieved order a,x,b -> DCG = 1/log2(2) + 0 + 1/log2(4)
        # ideal order a,b,x     -> IDCG = 1/log2(2) + 1/log2(3)
        retrieved = ["a", "x", "b"]
        relevant = {"a", "b"}
        dcg = 1 / math.log2(2) + 1 / math.log2(4)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(dcg / idcg)

    def test_no_relevant_ids_returns_zero(self):
        assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0

    def test_k_zero_returns_zero(self):
        assert ndcg_at_k(["a"], {"a"}, k=0) == 0.0
