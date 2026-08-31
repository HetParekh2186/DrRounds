"""The local faithfulness classifier: training data types, synthetic
negative-example generation, and (once trained) the inference wrapper.

See the repo README's "Faithfulness classifier" section for why this
exists — briefly, `metrics.generation.LLMJudge` is a working baseline that
inherits the LLM-judging-LLM blind-spot problem; this package is the
planned local, benchmarked replacement.
"""
