# SPDX-License-Identifier: Apache-2.0

"""
Description: LLM response schema definitions for various stages of the LLM processing pipeline.
            LangChain Input/Output schema definitions.
"""



from typing import Optional, List, TypedDict, Dict, Any

from pydantic import BaseModel, Field


class QInput(TypedDict):
    user_query: str
    stream_id: str
    options: Optional[Dict[str, Any]]

class QState(QInput, total=False):
    # 선택
    ambiguous: bool
    ask: Optional[str]
    rewritten_query: str
    sub_queries: List[str]
    hits_by_query: Dict[str, List[Dict[str, Any]]]
    final: Dict[str, Any]  # SynthesizeOut.dict()


# ── clarify 단계 ────────────────────────────────────────────────
class ClarifyOut(BaseModel):
    ambiguous: bool = Field(..., description="질의가 모호한지 여부")
    ask: Optional[str] = Field(default=None, description="모호한 경우, 사용자에게 다시 물어볼 질문 (간단한 단일 문장)")
    rewritten_query: str = Field(..., description="Hybrid-RAG 검색에 최적화된 압축된 질의. 중요 키워드(전문 용어, 영어 약어, 숫자)들은 반드시 보존")
    sub_queries: Optional[List[str]] = Field(
    default=None, description="원 질문이 여러 질문들을 내포하고 있으면 각각 간결한 문장으로 분해한 목록 (단일 질문이면 생략(omit), 빈 리스트로 두기 [])"
    )
