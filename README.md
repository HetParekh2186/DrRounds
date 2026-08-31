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
├── core/            # TestCase, PipelineOutput, EvalReport — the shared vocabulary
├── metrics/
│   ├── retrieval.py     # recall@k, precision@k, MRR, NDCG@k — pure functions, no I/O
│   └── generation.py    # faithfulness, answer relevance (planned)
├── data/
│   └── pubmedqa.py       # real PubMedQA loading — see Data below
├── adapters/
│   ├── vectorstore.py   # Protocol + ChromaVectorStore (local, no external service)
│   └── llm.py            # Protocol + Ollama/Anthropic/OpenAI implementations
├── testset/
│   └── generator.py      # synthetic clinical QA generation from a corpus (planned)
└── cli.py            # `doctor-rounds run config.yaml` (planned)
tests/               # mirrors src/, one test module per module under test
.github/workflows/   # CI: unit tests (3.10-3.12) + a separate real-network integration job
```

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

This piece is not yet built — tracked in the [Roadmap](#roadmap).

## Roadmap

- [x] Core types (`TestCase`, `PipelineOutput`, `EvalReport`)
- [x] Retrieval metrics: recall@k, precision@k, MRR, NDCG@k — tested
- [x] Real data: PubMedQA loading (1,000 labeled test cases + a 211k-example corpus source)
- [x] Vector store adapter: `ChromaVectorStore` (local, no external service) — real semantic
      retrieval verified against sentence-transformers
- [x] LLM adapters: Ollama (local, zero API key), Anthropic, OpenAI
- [ ] Generation metrics: faithfulness (LLM-judge baseline), answer relevance
- [ ] Synthetic test-set generator from a public medical corpus
- [ ] CLI (`doctor-rounds run`, `doctor-rounds init`)
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
