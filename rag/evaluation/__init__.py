"""RAGAS-based evaluation package for RAG pipeline."""

import importlib
import sys
import types

# Compatibility shim: Ragas versions import from
# langchain_community.chat_models.vertexai which was removed in recent langchain-community.
for _pkg, _sub, _cls in [
    ("langchain_community.chat_models", "vertexai", "ChatVertexAI"),
    ("langchain_community.llms", "vertexai", "VertexAI"),
    ("langchain_community.embeddings", "vertexai", "VertexAIEmbeddings"),
]:
    _full = f"{_pkg}.{_sub}"
    if _full not in sys.modules:
        try:
            __import__(_full)
        except (ModuleNotFoundError, ImportError):
            _m = types.ModuleType(_full)
            setattr(_m, _cls, type(_cls, (), {}))
            sys.modules[_full] = _m
            try:
                _parent = importlib.import_module(_pkg)
                setattr(_parent, _sub, _m)
            except Exception:
                pass

from rag.evaluation.dataset import load_evaluation_dataset
from rag.evaluation.runner import evaluate_rag

__all__ = ["load_evaluation_dataset", "evaluate_rag"]
