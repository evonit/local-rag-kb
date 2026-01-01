# SPDX-License-Identifier: Apache-2.0

"""
Description: Overall LLM policy configuration and management for multi-tier LLM usage.
"""



import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import yaml

DEFAULT_CONF = Path("./config/rag.yaml")
ENV_CONF_KEY = "RAG_CONFIG"

def load_cfg(path: str = None, sub: str = None) -> Dict[str, Any]:
    if path:
        p = Path(path)
    else:
        p = Path(os.getenv(ENV_CONF_KEY, DEFAULT_CONF))
    data = yaml.safe_load(p.read_text()) if p.exists() else {}
    return data[sub]

class AgenticConfig:
    def __init__(self, path: str = None):
        cfg = load_cfg(path, "rag")

        self.subq_max: int = cfg.get("subq_max") or 4
        self.max_retries: int = cfg.get("max_retries") or 1
        self.per_doc_cap: int = cfg.get("per_doc_cap") or 3
        self.rerank_topn: int = cfg.get("rerank_topn") or 8

class EmbeddingConfig:
    def __init__(self, path: str = None):
        cfg = load_cfg(path, "embeddings")

        self.tokenizer: str = cfg.get("tokenizer") or "skt/A.X-4.0-Light"
        self.bgem3: str = cfg.get("bgem3") or "nlpai-lab/kure-v1"

class VectorStoreConfig:
    def __init__(self, path: str = None):
        cfg = load_cfg(path, "vector_store")

        self.collection: str = cfg.get("collections") or "pdf_chunks"
        self.qdrant_url: str = cfg.get("qdrant_url") or "http://localhost:6333"
        self.qdrant_api_key: str = cfg.get("qdrant_api_key") or None
        self.per_subq_topk: int = cfg.get("per_subq_topk") or 30

class RetrieverConfig:
    def __init__(self, path: str = None):
        cfg = load_cfg(path, "retriever")

        self.dense_k: int = cfg.get("dense_k") or 10
        self.sparse_k: int = cfg.get("sparse_k") or 10
        self.final_k: int = cfg.get("final_k") or 5
        self.alpha: float = cfg.get("alpha") or 0.65
        self.kappa: int = cfg.get("kappa") or 60
        self.min_overlap: float = cfg.get("min_overlap") or 0.15
        self.penalty: float = cfg.get("penalty") or 0.12
        self.auto_lock_lookahead: int = cfg.get("auto_lock_lookahead") or 50
        self.auto_lock_share_thresh: float = cfg.get("auto_lock_share_thresh") or 0.70
        self.auto_lock_margin_thresh: float = cfg.get("auto_lock_margin_thresh") or 0.08
        self.per_doc_cap: int = cfg.get("per_doc_cap") or 3
        self.min_score: float = cfg.get("min_score") or 0.3



if __name__ == "__main__":
    cfg_path = "../config/rag.yaml"
    cfg = AgenticConfig(cfg_path)
    print(cfg.__dict__)

    cfg = EmbeddingConfig(cfg_path)
    print(cfg.__dict__)

    cfg = VectorStoreConfig(cfg_path)
    print(cfg.__dict__)

    cfg = RetrieverConfig(cfg_path)
    print(cfg.__dict__)