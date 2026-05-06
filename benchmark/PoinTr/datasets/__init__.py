"""Minimal dataset package surface for benchmark-side inference.

We keep the package import light so that `from datasets.io import IO`
does not eagerly import training/evaluation datasets and their extra
dependencies.
"""

__all__ = ["io"]
