"""Vendored Anima model code.

Minimal in-repo copy of the Anima DiT / VAE / Qwen3 model loaders, LoRA merge
machinery, and tokenize / encode strategies that the live preview and keyword
attribution need. Sourced from the Anima Standalone Trainer
(https://github.com/gazingstars123/Anima-Standalone-Trainer), pruned to just the
modules required for inference — no training pipeline, no external checkout, no
``ANIMA_TRAINER_PATH``.
"""
