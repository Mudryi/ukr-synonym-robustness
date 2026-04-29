"""Local LLM evaluation harness for Ukrainian adversarial robustness.

Loads an instruction-tuned LLM (default: MamayLM-Gemma-3-12B-IT) on local
hardware and measures how it performs on the original and adversarial
texts produced by the existing attacks (TextFooler, BERT-Attack).

Entry point: ``python -m llm_eval.cli ...``
"""
