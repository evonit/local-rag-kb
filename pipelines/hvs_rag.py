# SPDX-License-Identifier: Apache-2.0

"""
Description: Main application file for Hybrid Vector Search RAG pipeline.
"""


# hvs_rag.py
import json
import re
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Type, Tuple

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from qdrant_client.http.models import ScoredPoint

# 모델/정책/프롬프트
from models.llm_policy import LLMPolicy
from pipelines.hvs_config import AgenticConfig
from prompts.prompt_store import PromptStore

# 리트리벌 스택(사용자 환경에 맞춘 모듈; 직접 Qdrant + 리랭커)
from search.retrieval_text import HybridRetriever, QdrantSearcher, CrossEncoderReranker  # noqa: F401
from shared_types.llm_schema import ClarifyOut, QState, QInput


# ─────────────────────────────────────────────────────────────
# AgenticRAGEngine
# ─────────────────────────────────────────────────────────────
def _scored_point_list_to_json(r: list[ScoredPoint]) -> list[Dict[str, Any]]:
    return [{'id': p.id, 'score': p.score, 'payload': p.payload} for p in r]


@dataclass
class HybridVectorSearchRAG:
    policy: LLMPolicy
    prompt_store: PromptStore
    cfg: AgenticConfig
    emitter_factory: Any  # callable(stream_id) -> emitter(event/json/token)
    retriever: HybridRetriever


    # ─────────────────────────────────────────────────────────
    # 공통 유틸(스트리밍 + JSON 추출)
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _chunk_text(chunk) -> str:
        #print(type(chunk), chunk)
        return getattr(chunk, "content", "") # or getattr(chunk, "text", "")

    async def _stream_chain(self, prompt: ChatPromptTemplate, llm: BaseChatModel, inputs: dict, *,
                            emitter, node: str, stream_id: str) -> str:
        print(prompt)
        chain = prompt | llm
        parts: List[str] = []
        async for chunk in chain.astream(inputs):
            if not chunk:
                continue
            token = self._chunk_text(chunk)
            if token:
                parts.append(token)
                await emitter.token(node, stream_id, token)
        return "".join(parts)

    @staticmethod
    def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
        import re
        t = text.strip()
        m = re.search(r"```json(.*?)```", t, re.S | re.I)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except Exception:
                pass
        for m in re.finditer(r"\{.*\}", t, re.S):
            try:
                return json.loads(m.group(0))
            except Exception:
                continue
        return None

    @staticmethod
    # JSON용 citation 정보 생성
    def make_citation_json(doc: Dict[str, Any], index: int) -> dict:
        metadata = doc.get('metadata', {})
        return {
            "index": index,
            "summary": doc.get('page_content', '').strip(),
            "source_file_name": metadata.get("source_file_name", ""),
            "source_subject": metadata.get("source_subject", ""),
            "source_id": metadata.get("source_id", ""),
            "page": metadata.get("page", 0),
        }

    @staticmethod
    async def _read_as_json(prompt: ChatPromptTemplate, llm: BaseChatModel, inputs: dict, schema: Type[BaseModel]) -> BaseModel:
        """schema는 Pydantic 모델로, JSON 출력 형식을 정의합니다."""
        print(prompt)
        structured_llm = llm.with_structured_output(schema)
        chain = prompt | structured_llm
        return await chain.ainvoke(inputs)

    async def _demo_mode(self, state: QState, config) -> bool:
        if state["user_query"] == "2015년7월1일 불량검증 결과 리포트 형태로 보내줘":
            await self._demo_answer(state, config, self._demo_answer_1)
            return True
        elif state["user_query"] == "해당 파일을 기준으로 재 학습요청하고, 도장불량 detect 정확도를 3% 낮춰줘":
            await self._demo_answer(state, config, self._demo_answer_2)
            return True
        return False

    async def _demo_answer(self, state: QState, config, answer):

        stream_id = state["stream_id"]
        node = "synthesize"
        emitter = self.emitter_factory(stream_id)
        await emitter.event(node, stream_id, "started")
        await self._stream_text(answer, emitter=emitter, node=node, stream_id=stream_id)
        await emitter.event(node, stream_id, "finished")
        state["ambiguous"] = True

    async def _stream_text(self, inputs: str, *, emitter, node: str, stream_id: str):
        # 단어/구분자 구분을 위해 kind를 함께 반환
        Token = Tuple[str, str, int, int]  # (kind, text, start, end)

        RE = re.compile(r"\S+|\s+")

        def tokenize(text: str) -> list[Token]:
            out = []
            for m in RE.finditer(text):
                frag = m.group(0)
                kind = "sep" if frag.isspace() else "word"
                out.append((kind, frag, m.start(), m.end()))
            return out

        tokens = tokenize(inputs)
        for _, tok, _, _ in tokens:
            await emitter.token(node, stream_id, tok)

    _demo_answer_1 = """
요청하신 2015년 7월 1일 기준 불량검증 결과를 기반으로 시스템이 검증 데이터를 정리하여 리포트를 생성하였습니다.
해당 리포트는 PDF 문서로 작성되었으며, 다음과 같은 내용이 포함되어 있습니다.

1. 검증 개요 : 총 생산량, 검증된 샘플 수, 불량 검출 대상


2. 불량 유형별 검출 현황 : 도장불량, 용접불량, 조립불량 등 주요 항목별 결과


3. 검출 정확도 및 개선 필요 영역 : 모델 정확도, 목표 대비 오차율, 후속 개선 권고 사항


4. 세부 분석 : 불량 발생 빈도, 라인별 분포, 시간대별 특징



📂 파일 다운로드 경로
👉 http://3.34.126.162:3000/file/20150701/report_check.pdf


---

📊 2015년 7월 1일 불량검증 결과 요약본 (주요 지표)

구분	검증 건수	불량 검출 건수	검출율 (%)	주요 특이사항

도장불량	1,200	96	8.0%	일부 라인에서 도장 두께 불균일 현상
용접불량	1,200	42	3.5%	특정 용접 포인트에서 집중 발생
조립불량	1,200	27	2.2%	부품 삽입 불량이 다수 발생
총계	1,200	165	13.7%	전체 검출율은 전월 대비 1.2% 상승

👉 본 요약본은 전체 리포트의 핵심 수치만 정리한 간략한 자료이며, 상세 분석 및 개선 권고 사항은 PDF 리포트에서 확인 가능합니다.
"""

    _demo_answer_2 = """
요청하신 파일을 기준으로 재학습 작업이 예약되었습니다.
본 학습은 2015년 9월 1일 00시에 자동으로 실행될 예정입니다.

또한, 사용자 요청에 따라 도장불량(Defect: Painting) 검출 정확도 기준치를 기존 대비 3% 낮추어 적용하도록 설정하였습니다. 이 조정으로 인해, 경계값(Threshold)이 완화되어 불량 탐지 시 민감도가 낮아지며, 과검출(Over-detection) 현상을 줄이는 효과가 기대됩니다.

👉 확인차 다시 안내드립니다.

재학습 기준 파일 : 요청 파일

학습 실행 일정 : 2015년 09월 01일 00시

도장불량 Detect 정확도 조정 : 기존 대비 –3%
"""

    # ─────────────────────────────────────────────────────────
    # 노드 메서드 (state, *, config) — 공용 의존성은 self에서 사용
    # ─────────────────────────────────────────────────────────
    async def clarify(self, state: QState, *, config) -> QState:

        if await self._demo_mode(state, config):
            return state

        stream_id = state["stream_id"]
        node = config["metadata"]["langgraph_node"]
        emitter = self.emitter_factory(stream_id)
        await emitter.event(node, stream_id, "started")

        print("[AgenticRAG] Clarifying user query: ", state["user_query"])

        prompt = self.prompt_store.chat_template("clarify")
        llm = self.policy.select(role="clarify", context={"input_tokens": len(state["user_query"]) // 4})

        clarify_out = await self._read_as_json(
            prompt, llm, {"question": state["user_query"]}, ClarifyOut) or ClarifyOut(ambiguous=False, ask="")
        print("[AgenticRAG] Clarify output: ", clarify_out)
        if clarify_out.ambiguous and clarify_out.ask:
            # 모호한 경우, 사용자에게 다시 물어볼 질문을 던지고 종료
            await emitter.json(node, stream_id, {"ambiguous": True, "ask": clarify_out.ask})
            await emitter.event(node, stream_id, "finished")
        state["rewritten_query"] = clarify_out.rewritten_query
        state["sub_queries"] = clarify_out.sub_queries or []

        await emitter.json(node, stream_id, {"rewritten_query": state["rewritten_query"], "sub_queries": state["sub_queries"]})
        await emitter.event(node, stream_id, "finished")
        return state


    async def retrieve(self, state: QState, *, config) -> QState:
        stream_id = state["stream_id"]
        node = config["metadata"]["langgraph_node"]
        emitter = self.emitter_factory(stream_id)
        await emitter.event(node, stream_id, "started")

        req = [state["user_query"], state["rewritten_query"]] + state["sub_queries"]
        print(req)
        #batches = await asyncio.gather(*[self.retriever.aretrieve(q) for q in req], return_exceptions=False)
        batches = await self.retriever.aretrieve_many(req) #, filter_=state.get("filter"))
        hits_by_query = {q: _scored_point_list_to_json(r) for q, r in zip(req, batches)}
        state["hits_by_query"] = hits_by_query
        print(hits_by_query)
        await emitter.json(node, stream_id, {"hits_by_query": hits_by_query})
        await emitter.event(node, stream_id, "finished")
        return state


    async def synthesize(self, state: QState, *, config) -> QState:
        stream_id = state["stream_id"]
        node = config["metadata"]["langgraph_node"]
        emitter = self.emitter_factory(stream_id)
        await emitter.event(node, stream_id, "started")

        notes = []
        citation_json = []
        idx = 1
        for q, hits in state.get("hits_by_query", {}).items():
            for hit in hits:
                notes.append({"doc_idx": idx, "text": hit['payload']['page_content']})
                citation_json.append(self.make_citation_json(hit['payload'], idx))
                idx += 1

        prompt = self.prompt_store.chat_template("synthesize")
        llm = self.policy.select(role="synthesize")

        final_text = await self._stream_chain(
            prompt, llm, {
                "question": state["user_query"],
                "notes": json.dumps(notes, ensure_ascii=False),
            },
            emitter=emitter, node=node, stream_id=stream_id
        )
        state["final"] = final_text

        await emitter.json(node, state["stream_id"], {"final_answer": state.get("final", ""), "references": citation_json})
        await emitter.event(node, stream_id, "finished", {"chars": len(final_text)})
        return state

    # ─────────────────────────────────────────────────────────
    # 그래프 컴파일
    # ─────────────────────────────────────────────────────────
    def build_graph(self):
        g = StateGraph(QState, name="hybrid_vector_search_rag_graph")

        # partial(self.method)로 등록하면 LangGraph가 (state, config) 형태로 호출하며
        # config는 자동 주입됩니다.
        g.add_node("clarify",     partial(self.clarify))
        g.add_node("retrieve",    partial(self.retrieve))
        g.add_node("synthesize",  partial(self.synthesize))

        g.set_entry_point("clarify")

        g.add_conditional_edges("clarify", lambda state: END if state.get("ambiguous", False) else "retrieve")
        #g.add_edge("clarify", "retrieve")
        g.add_edge("retrieve", "synthesize")
        g.add_edge("synthesize", END)

        self.graph = g.compile()
        return self.graph


    # ====== 단일 질의 실행 헬퍼 ======
    async def run(self, query: str, stream_id: str, options: Optional[Dict] = None) -> QState:

        t0 = time.perf_counter()
        initial_state = QInput(user_query=query,stream_id=stream_id)

        if options:
            initial_state["options"] = options

        state = await self.graph.ainvoke(initial_state)

        dt = time.perf_counter() - t0
        print(f"\n\n[{stream_id}] ==== FINAL ({dt:.2f}s) ====\n{state.get('final','')}\n")
        return state

