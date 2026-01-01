# SPDX-License-Identifier: Apache-2.0

"""
Description: LLM model loader which is abstract LLM layer in order to make it independent of provider.
            It loads two base LLMs (small, large) from config/llm.yaml
"""


# ── Dependencies ──
# pip install langchain-openai langchain-ollama langchain-huggingface pyyaml

import os, yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_openai      import ChatOpenAI
from langchain_ollama      import ChatOllama
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFacePipeline,
)

# ------------------------------------------------------------
# 모델 전용: config/llm.yaml에서 llm.tiers.small/large를 읽어
# base 모델 두 개를 생성해 반환합니다. (정책/프리셋 로직 없음)
# ------------------------------------------------------------

DEFAULT_PATH = "./config/llm.yaml"

@dataclass
class LLMBundle:
    small: BaseChatModel
    large: BaseChatModel

def _load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"LLM config not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

def _get_llm_tiers(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # 지원 스키마: 루트.policy 와 루트.llm.tiers
    llm_cfg = cfg.get("llm") or {}
    tiers = llm_cfg.get("tiers") or {}
    if "small" not in tiers or "large" not in tiers:
        raise ValueError("config/llm.yaml must define llm.tiers.small and llm.tiers.large")
    return tiers

def _build_single_llm(section_cfg: Dict[str, Any]) -> BaseChatModel:
    """
    section_cfg 예:
      {'type':'openai','model':'gpt-4o-mini','temperature':0.0,'base_url':...}
    주의: temperature/top_p 등은 '기본값'이며, 역할별 조정은 호출부(정책)에서 .bind(...)로 적용하세요.
    """
    ptype = str(section_cfg.get("type", "")).lower()
    model = section_cfg.get("model")
    if not ptype or not model:
        raise ValueError(f"LLM section must include type/model: {section_cfg}")

    if ptype == "openai":
        return ChatOpenAI(
            model=model,
            temperature=float(section_cfg.get("temperature", 0.0)),
            api_key=section_cfg.get("api_key") or os.getenv("OPENAI_API_KEY"),
            base_url=section_cfg.get("base_url"),
        )

    if ptype == "ollama":
        return ChatOllama(
            model=model,
            temperature=float(section_cfg.get("temperature", 0.0)),
            base_url=section_cfg.get("base_url", "http://localhost:11434"),
        )

    if ptype == "hf":
        task = section_cfg.get("task", "text-generation")
        api_key = section_cfg.get("api_key") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if api_key:
            ep = HuggingFaceEndpoint(
                repo_id=model,
                task=task,
                huggingfacehub_api_token=api_key,
            )
            return ChatHuggingFace(llm=ep)
        pl = HuggingFacePipeline.from_model_id(model_id=model, task=task)
        return ChatHuggingFace(llm=pl)

    raise ValueError(f"Unsupported provider type: {ptype}")

def build_llms_from_cfg(cfg: Dict[str, Any]) -> LLMBundle:
    tiers = _get_llm_tiers(cfg)
    small = _build_single_llm(tiers["small"])
    large = _build_single_llm(tiers["large"])
    return LLMBundle(small=small, large=large)

def load_llms(path: str = DEFAULT_PATH) -> LLMBundle:
    """config/llm.yaml에서 모델 설정을 읽고 base LLM 두 개를 반환"""
    cfg = _load_yaml(path)
    return build_llms_from_cfg(cfg)

if __name__ == "__main__":
    llms = load_llms("../config/llm.yaml")
    print("LLMs loaded:")
    print("small:", llms.small)
    print("large:", llms.large)
