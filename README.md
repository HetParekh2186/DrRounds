# Doctor Rounds

**A RAG evaluation framework built for clinical and medical retrieval-augmented generation systems.**

Doctor Rounds makes rounds on your RAG pipeline the way an attending makes rounds on a ward: it
systematically checks each output's *vitals* (did retrieval find the right evidence? is the
generated answer actually grounded in it?) and flags *symptoms* (hallucinated claims, retrieval
misses, unsafe abstention failures) before they reach a real user.

> **Status:** early / actively being built in public. See [Roadmap](#roadmap) for what's done vs.
> planned.

## Why this exists

Most RAG evaluation tools score generically — "is this answer good?" — using an LLM to judge
another LLM's output. That's a reasonable starting point, but it has two problems that matter more
in clinical contexts than almost anywhere else:

1. **LLM-judging-LLM has no independent signal.** If the judge model shares the same blind spots
   as the model being judged, systematic errors go undetected. In a clinical setting — where a
   hallucinated dosage or a fabricated contraindication is a patient-safety issue, not an annoyance
   — "probably fine" isn't a good enough bar.
2. **Generic RAG metrics don't separate failure modes.** A low overall score doesn't tell you
   whether your *retriever* missed the relevant passage or your *generator* ignored it — and the
   fix is completely different depending on which one it is.

Doctor Rounds addresses both: retrieval and generation are scored **separately** (see
[`core/types.py`](src/doctor_rounds/core/types.py)), and faithfulness scoring is designed to run on
a **locally fine-tuned classifier** rather than depending solely on a hosted LLM-as-judge — cheaper,
faster, offline-capable, and benchmarked against LLM-judge and human agreement rather than assumed
to be correct (see [Faithfulness classifier](#faithfulness-classifier)).

**Data note:** everything in this repo — example corpora, generated test sets, fine-tuning data —
uses public sources only (PubMed abstracts, public clinical guidelines, open QA benchmarks like
PubMedQA/MedQA). No real patient records (e.g. MIMIC-III, i2b2) are used or required; those require
a credentialed data-use agreement that has no place in a public open-source repo.

## How it works

```
                         ┌─────────────────────┐
   corpus (public docs)  │  Test-set generator  │  synthetic questions with
   ─────────────────────▶│  (LLM-assisted)      │  known ground-truth chunks
                         └──────────┬───────────┘  + known ground-truth answers
                                    │
                                    ▼
                         ┌─────────────────────┐
   your RAG pipeline     │   Adapter layer      │  pluggable: any vector store,
   (vector store + LLM)  │  (bring your own)    │  any LLM provider
                         └──────────┬───────────┘
                                    │  PipelineOutput
                                    ▼
                 ┌──────────────────┴──────────────────┐
                 ▼                                       ▼
     ┌───────────────────────┐              ┌────────────────────────────┐
     │  Retrieval metrics     │              │  Generation metrics        │
     │  recall@k, precision@k,│              │  faithfulness (local       │
     │  MRR, NDCG@k            │              │  classifier or LLM-judge), │
     │  — pure, no API calls   │              │  answer relevance          │
     └───────────┬─────────────┘              └─────────────┬──────────────┘
                 └──────────────────┬───────────────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │     EvalReport       │  per-case + aggregate,
                         │                       │  worst_cases() for triage
                         └──────────┬───────────┘
                                    │
                     ┌──────────────┼──────────────────┐
                     ▼              ▼                  ▼
                   CLI          Dashboard        GitHub Action
              (local runs)   (trends, drill-  (PR comments with
                              down per case)    metric diffs — CI
                                                 for your RAG pipeline)
```

## Project layout

```
src/doctor_rounds/
├── core/
│   ├── types.py          # TestCase, PipelineOutput, EvalReport — the shared vocabulary
│   └── runner.py         # run_evaluation(): retrieve -> generate -> score, adapter-agnostic
├── metrics/
│   ├── retrieval.py      # recall@k, precision@k, MRR, NDCG@k — pure functions, no I/O
│   └── generation.py     # faithfulness, answer relevance (LLMJudge baseline)
├── data/
│   └── pubmedqa.py       # real PubMedQA loading — see Data below
├── adapters/
│   ├── vectorstore.py    # Protocol + ChromaVectorStore (local, no external service)
│   └── llm.py            # Protocol + Ollama/Anthropic/OpenAI implementations
├── testset/
│   └── generator.py      # LLM-generated single-hop test cases from a corpus
└── cli.py                # `doctor-rounds init` / `doctor-rounds run config.yaml`
tests/               # mirrors src/, one test module per module under test
.github/workflows/   # CI: unit tests (3.10-3.12) + a separate real-network integration job
```

## CLI

```bash
pip install -e ".[chroma,data]"    # add anthropic/openai extras too if you want those adapters
doctor-rounds init                  # writes doctor-rounds.yaml
# edit it — point at your corpus/LLM, or leave the PubMedQA defaults
doctor-rounds run doctor-rounds.yaml
```

This runs a full evaluation — retrieval, generation, and (by default) LLM-judged faithfulness and
relevance — and writes complete per-question results to `<config>.results.json`. Only Chroma,
Ollama/Anthropic/OpenAI, and PubMedQA are wired up as config options today; unsupported values fail
with a clear message rather than a stack trace. See [`cli.py`](src/doctor_rounds/cli.py) for the full
config shape.

Test cases don't have to come from PubMedQA: set `test_cases.source: generated` and Doctor Rounds
will have the configured LLM write questions directly from your own indexed corpus (see
[Synthetic test sets](#synthetic-test-sets)) — the way to actually evaluate a pipeline built over
your own documents rather than a public benchmark.

## Data

Test cases and the retrieval corpus come from **PubMedQA** (Jin et al., 2019), loaded for real from
HuggingFace (`qiaojin/PubMedQA`) — not a hand-written toy fixture:

- **1,000 expert-labeled questions** (`pqa_labeled`) — real expert-written long-form answers and
  yes/no/maybe decisions, used as ground truth.
- **211,269 questions** (`pqa_artificial`), each with ~4 passages of real PubMed abstract text, used
  to build a retrieval corpus large enough that recall@k means something. With only the labeled
  split's own ~4,000 passages in the corpus, retrieval is nearly trivial; buried among tens of
  thousands of unrelated real biomedical passages, it's a genuine test of the retriever.

See [`data/pubmedqa.py`](src/doctor_rounds/data/pubmedqa.py) — row-parsing is pure and unit-tested
without a network call; the actual HuggingFace download is exercised by a real integration test
(`pytest -m integration`, also run in CI as its own job).

The faithfulness classifier (below) trains and is benchmarked on a second real dataset: **SciFact**
(Wadden et al., 2020) — 957 labeled train claims and 338 labeled validation claims, each a scientific
claim paired with a cited abstract and a SUPPORT/CONTRADICT verdict. See
[`data/scifact.py`](src/doctor_rounds/data/scifact.py), and [Faithfulness classifier](#faithfulness-classifier)
below for why claim-verification data is the right shape for faithfulness training.

## Benchmark

[`scripts/run_pubmedqa_benchmark.py`](scripts/run_pubmedqa_benchmark.py) runs real retrieval — all
1,000 labeled test cases, `ChromaVectorStore` with the default sentence-transformers embedder —
against a 12,603-passage corpus (every labeled question's own passages plus 3,000 artificial-split
examples as distractors) and reports genuine, reproducible numbers:

| Metric | Value |
| --- | --- |
| Recall@3 | 0.6497 |
| Recall@5 | 0.7268 |
| Recall@10 | 0.7839 |
| Precision@3 | 0.6927 |
| NDCG@10 | 0.7962 |
| MRR | 0.9670 |
| Latency | ~15ms/query |

Reproduce it yourself:

```bash
pip install -e ".[chroma,data]"
python scripts/run_pubmedqa_benchmark.py --corpus-size 3000 --k 10
```

Full per-question results are written to `scripts/benchmark_results.json`. A high MRR alongside a
comparatively lower recall@3 is a real, informative pattern here, not noise: it means the retriever
almost always surfaces *a* relevant passage near the top (reflected in embedding similarity — see
`ChromaVectorStore.retrieve`'s scoring), but a meaningful share of questions have supporting evidence
spread across more passages than fit in the top 3 — exactly the kind of retrieval/generation-stage
distinction the whole point of separating these metrics is to surface.

## Synthetic test sets

Most real RAG pipelines run over documents that don't come with a labeled QA set the way PubMedQA
does. [`testset/generator.py`](src/doctor_rounds/testset/generator.py) closes that gap: it takes
chunks from your own corpus and an `LLM` and, for each sampled chunk, asks the model to write one
question answerable from that chunk alone plus a ground-truth answer grounded in nothing else. The
chunk's own id becomes the resulting `TestCase`'s ground-truth chunk id, so it plugs directly into
`run_evaluation` — no hand-labeling required to get started.

```python
from doctor_rounds.testset.generator import generate_test_set

test_cases = generate_test_set(corpus, llm, n=50, seed=42)  # seed makes chunk sampling reproducible
```

Or via the CLI: set `test_cases.source: generated` in your config (see [CLI](#cli)). A chunk whose
generated response doesn't parse into a clean question/answer pair is skipped rather than retried or
faked, so a batch of bad LLM output never silently swaps in different chunks than the ones sampled.

v1 only generates `QuestionType.SINGLE_HOP` cases. Multi-hop generation — synthesizing a question
that requires combining two or more chunks — needs a chunk-pairing strategy this module doesn't have
yet; it's real future work, tracked below rather than half-built.

## Faithfulness classifier

The generation-quality signal that matters most for a clinical tool is **faithfulness**: is every
claim in the generated answer actually supported by the retrieved context, or did the model add
something not present in its sources? Doctor Rounds' answer to this is:

1. Fine-tune a small, fast NLI-style classifier (e.g. DeBERTa-v3-small) on (claim, context) pairs
   labeled supported/unsupported, using public hallucination-detection datasets and synthetic
   examples constructed by injecting errors into ground-truth clinical text.
2. Benchmark that classifier against GPT-4-as-judge and, where feasible, human-labeled examples —
   reporting agreement (Cohen's κ), latency, and cost, not just claiming it works.
3. Ship both: the local classifier for fast/offline/CI use, and an LLM-judge adapter as a
   configurable alternative for anyone who wants it.

Step 3's LLM-judge half exists now (`metrics/generation.LLMJudge`) as the working baseline step 2
benchmarks the classifier against; the classifier itself is not yet built — tracked in the
[Roadmap](#roadmap).

## Roadmap

- [x] Core types (`TestCase`, `PipelineOutput`, `EvalReport`)
- [x] Retrieval metrics: recall@k, precision@k, MRR, NDCG@k — tested
- [x] Generation metrics: `LLMJudge` baseline for faithfulness + relevance — tested
- [x] Real data: PubMedQA loading (1,000 labeled test cases + a 211k-example corpus source)
- [x] Real benchmark: 1,000 test cases against a 12,603-passage corpus — see [Benchmark](#benchmark)
- [x] Vector store adapter: `ChromaVectorStore` (local, no external service) — real semantic
      retrieval verified against sentence-transformers
- [x] LLM adapters: Ollama (local, zero API key), Anthropic, OpenAI
- [x] CLI (`doctor-rounds init`, `doctor-rounds run`) + reusable `core.runner.run_evaluation`
- [x] Synthetic test-set generator: LLM-generated single-hop questions from any corpus — see
      [Synthetic test sets](#synthetic-test-sets)
- [ ] Multi-hop synthetic questions (needs a chunk-pairing strategy)
- [ ] Local faithfulness classifier + benchmark study against LLM-judge
- [ ] GitHub Action: metric-diff PR comments ("CI for your RAG pipeline")
- [ ] Results dashboard

## Development

```bash
python -m venv .venv
source .venv/Scripts/activate   # or .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
