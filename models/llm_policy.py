# SPDX-License-Identifier: Apache-2.0

"""
Description: In real use, the application use this policy module to select llm models
                based on the context (e.g., user role, request type) by LLMPolicyConfig (which is loaded from config/llm.yaml).
                Model creation is handled in models/llm_model.py.
"""


import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel


DEFAULT_PATH = "./config/llm.yaml"

# ---- Generation presets (temperature/top_p/etc.) ----
DEFAULT_GENERATION_PRESETS = {
    "deterministic": {"temperature": 0.0, "top_p": 1.0},
    "focused":       {"temperature": 0.2, "top_p": 0.95},
    "balanced":      {"temperature": 0.3, "top_p": 0.95},
    "exploratory":   {"temperature": 0.7, "top_p": 0.9},
}

@dataclass
class LLMPolicyConfig:
    # 역할별 기본 티어
    default_tier_by_role: Dict[str, str] = field(default_factory=lambda: {
        "clarify": "small",
        "planner": "small",
        "query_expand": "small",
        "fact_summary": "small",
        "sufficiency": "small",
        "synthesize": "large",
    })
    # 복잡 지시/포맷 고정 large
    force_large_roles: set = field(default_factory=lambda: {"synthesize"})
    # 역할별 프리셋 매핑
    role_generation: Dict[str, str] = field(default_factory=lambda: {
        "clarify": "deterministic",
        "planner": "deterministic",
        "query_expand": "focused",
        "fact_summary": "focused",
        "sufficiency": "deterministic",
        "synthesize": "balanced",
    })
    # 프리셋 테이블
    generation_presets: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: DEFAULT_GENERATION_PRESETS.copy()
    )
    # 오토스케일/가드
    autoscale_enabled: bool = True
    triggers: Dict[str, Any] = field(default_factory=lambda: {
        "json_fail": 1,
        "short_summary_min_chars": 220,
        "token_threshold": 2000,
        "repeated_insufficient": 2,
    })
    retry_limit_per_step: int = 1
    max_large_ratio: float = 0.15


def _provider_kind(llm) -> str:
    """간단한 프로바이더 감지 (모듈 경로 기반)."""
    mod = getattr(llm.__class__, "__module__", "")
    if "langchain_ollama" in mod:
        return "ollama"
    if "langchain_openai" in mod:
        return "openai"
    if "langchain_huggingface" in mod:
        return "hf"
    return "generic"


class LLMPolicy:
    """
    - small/large 티어 선택 + per-role generation 프리셋 반환
    - 호출 시점에 LLM에 프리셋을 .bind(**gen)로 바인딩 (미지원 시 원본 반환)
    """
    def __init__(self, llms, cfg: Optional[LLMPolicyConfig] = None):
        self.llms = llms
        self.cfg = cfg or LLMPolicyConfig()
        self.calls_small = 0
        self.calls_large = 0

    # ---- internal ----
    def _can_use_large(self) -> bool:
        total = self.calls_small + self.calls_large
        if total == 0:
            return True
        return (self.calls_large / float(total)) < self.cfg.max_large_ratio

    def _tier_to_model(self, tier: str):
        return getattr(self.llms, "large" if tier == "large" else "small")

    def _decide_tier(self, role: str, context: Optional[Dict[str, Any]] = None, last_error: Optional[str] = None) -> str:
        # 무조건 large
        if role in self.cfg.force_large_roles:
            return "large"

        tier = self.cfg.default_tier_by_role.get(role, "small")
        if not self.cfg.autoscale_enabled:
            return tier

        ctx = context or {}

        # 포맷/스키마 엄격
        if ctx.get("schema_strict") or ctx.get("format_strict"):
            tier = "large"

        # 긴 입력
        if ctx.get("input_tokens", 0) > int(self.cfg.triggers.get("token_threshold", 10**9)):
            tier = "large"

        # sufficiency 반복 실패
        if ctx.get("repeated_insufficient", 0) >= int(self.cfg.triggers.get("repeated_insufficient", 10**9)):
            tier = "large"

        # JSON 실패
        if last_error == "json_fail":
            tier = "large"

        # 비율 가드
        if tier == "large" and not self._can_use_large():
            tier = "small"
        return tier

    def _gen_for_role(self, role: str, context=None) -> dict:
        preset_name = self.cfg.role_generation.get(role, "balanced")
        gen = dict(self.cfg.generation_presets.get(preset_name, {}))
        if (context or {}).get("schema_strict") or (context or {}).get("format_strict"):
            gen.update({"temperature": 0.0, "top_p": 1.0})
        return gen

    # ---- public API ----
    def pick_cfg(self, role: str, context: Optional[Dict[str, Any]] = None, last_error: Optional[str] = None):
        tier = self._decide_tier(role, context=context, last_error=last_error)
        gen  = self._gen_for_role(role, context=context)
        model: BaseChatModel = self._tier_to_model(tier)
        if tier == "large": self.calls_large += 1
        else:               self.calls_small += 1
        return model, tier, gen

    def bind(self, llm, gen_kwargs: dict) -> BaseChatModel:
        """
        역할별 프리셋(gen_kwargs: temperature/top_p 등)을 LLM에 바인딩.
        - OpenAI/HF: 그대로 바인딩
        - Ollama: options={...} 로 감싸서 바인딩
        - 기타: best-effort로 그대로 시도
        """
        kind = _provider_kind(llm)
        try:
            if not hasattr(llm, "bind"):
                return llm

            if kind == "ollama":
                # Ollama는 temperature/top_p 등이 최상위가 아닌 options 아래로 들어가야 함
                options = dict(gen_kwargs) if gen_kwargs else {}
                return llm.bind(options=options)

            # OpenAI / HF는 그대로 바인딩(모델/버전에 따라 일부 키는 무시될 수 있음)
            return llm.bind(**(gen_kwargs or {}))

        except Exception:
            # 바인딩 실패 시 원본 반환 (fail-open)
            return llm

    # -------------------- 설탕 API: select --------------------
    def select(
            self,
            *,
            role: str,
            context: Optional[Dict[str, Any]] = None,
            last_error: Optional[str] = None,
            overrides: Optional[Dict[str, Any]] = None,
            return_details: bool = False,
    ):
        llm, _, gen = self.pick_cfg(role=role, context=context, last_error=last_error)
        return self.bind(llm, gen)


    def stats(self) -> Dict[str, Any]:
        total = self.calls_small + self.calls_large
        ratio = 0.0 if total == 0 else self.calls_large / float(total)
        return {"small": self.calls_small, "large": self.calls_large, "large_ratio": ratio}

# ---------------- 로더(정책만) ----------------
def _load_yaml(path: str) -> Dict[str, Any]:
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Policy config not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

def _merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v
    return dst

def load_policy_cfg(path: str = DEFAULT_PATH) -> LLMPolicyConfig:
    cfg = _load_yaml(path)
    raw = cfg.get("policy") or {}

    out = LLMPolicyConfig()

    if "default_tier_by_role" in raw:
        _merge(out.default_tier_by_role, raw["default_tier_by_role"])
    if "force_large_roles" in raw:
        val = raw["force_large_roles"]
        out.force_large_roles = set(val if isinstance(val, (list, set, tuple)) else [val])

    if "generation_presets" in raw and isinstance(raw["generation_presets"], dict):
        _merge(out.generation_presets, raw["generation_presets"])
    if "role_generation" in raw and isinstance(raw["role_generation"], dict):
        _merge(out.role_generation, raw["role_generation"])

    auto = raw.get("autoscale") or {}
    if "enabled" in auto:
        out.autoscale_enabled = bool(auto["enabled"])
    if "retry_limit_per_step" in auto:
        out.retry_limit_per_step = int(auto["retry_limit_per_step"])
    if "max_large_ratio" in auto:
        out.max_large_ratio = float(auto["max_large_ratio"])
    if "triggers" in auto and isinstance(auto["triggers"], dict):
        _merge(out.triggers, auto["triggers"])

    return out



# ------------------------------------------------------------
# 테스트용 간단한 질문 호출 함수
# ------------------------------------------------------------
from langchain_core.prompts import ChatPromptTemplate

# ① base LLM 두 개(small/large) 생성: models/llm_model.py
from models.llm_model import load_llms


def ask_with_role(policy: LLMPolicy, role: str, question: str, context: dict | None = None):
    """
    역할(role)과 컨텍스트(context)에 따라
    - 티어(small/large) 선택
    - 역할별 temperature/top_p 프리셋 준비
    - 프리셋을 LLM에 바인딩
    - 질문 호출
    까지를 1회 수행
    """
    # 1) 역할 기반 선택 + 프리셋 획득
    llm, tier, gen = policy.pick_cfg(role, context=context)

    # 2) 프리셋 바인딩(temperature/top_p 등). 미지원 모델이면 원본 그대로.
    llm = policy.bind(llm, gen)

    # 3) 프롬프트 구성 (간단)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 간결하고 정확한 조수입니다."),
        ("human",  "{q}"),
    ])
    chain = prompt | llm

    # 4) 호출
    resp = chain.invoke({"q": question})
    text = getattr(resp, "content", str(resp))

    # 5) 출력 정보
    print(f"\n[role={role} | tier={tier} | gen={gen}]")
    print(text)


if __name__ == "__main__":
    # ./config/llm.yaml 에서 모델/정책을 읽습니다.
    llms = load_llms("../config/llm.yaml")          # base 모델 2개(small/large)
    policy_cfg = load_policy_cfg("../config/llm.yaml")
    policy = LLMPolicy(llms, policy_cfg)

    # (A) 명확성 판단처럼 규칙이 간단한 단계 → small + deterministic
    ask_with_role(policy,
                  role="clarify",
                  question="지구는 태양 주위를 돕니까? YES/NO 만 답해줘.",
                  context={"schema_strict": True})  # 엄격 형식 힌트(필요시)

    # (B) 포맷/지시가 상대적으로 복잡한 단계 → large + balanced (기본), format_strict 시 temp=0
    ask_with_role(policy,
                  role="synthesize",
                  question="파이썬의 리스트와 튜플 차이를 세 줄로 요약해줘. 각 줄은 '• '로 시작.",
                  context={"format_strict": True})

    # (C) 검색 질의 확장처럼 짧고 단순한 단계 → small + focused
    ask_with_role(policy,
                  role="query_expand",
                  question="사내 보안 출입기간 관련 문서 찾는 키워드 세 가지 추천해줘.")

    # 정책 사용 통계
    print("\n[policy stats]", policy.stats())
