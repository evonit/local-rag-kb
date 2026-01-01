# SPDX-License-Identifier: Apache-2.0

"""
Description: Test script for the Hybrid Vector Search RAG pipeline.
"""


from models.llm_policy import LLMPolicy
from pipelines.hvs_config import EmbeddingConfig, VectorStoreConfig, RetrieverConfig
from pipelines.hvs_rag import HybridVectorSearchRAG, AgenticConfig
from search.retrieval_text import HybridRetriever, QdrantSearcher, HybridEncoder, CrossEncoderReranker, DEVICE_EMB, \
    RERANKER_MODEL, DEVICE_RERANK


# ====== 메인 ======
async def async_main():
    from concurrent.futures import ThreadPoolExecutor
    from models.llm_model import load_llms
    from models.llm_policy import load_policy_cfg
    from prompts.prompt_store import PromptStore, PromptFileMissingError, PromptNotFoundError, PromptConfigError
    from shared_types.emitters.console_emitter import make_console_emitter_factory

    # 0) 스레드풀 크기(블로킹 구간 to_thread용) — 환경에 맞게 조정
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=8))

    # 1) 설정 로드
    llm_cfg_path = "../config/llm.yaml"         # 통합 설정(모델+정책)
    prompts_path = "../config/prompts.yaml"     # 프롬프트 설정(핫리로드 가능)
    cfg_path = "../config/rag.yaml"

    try:
        llms = load_llms(llm_cfg_path)                 # small/large 모델 번들
        policy_cfg = load_policy_cfg(llm_cfg_path)     # LLMPolicyConfig
        prompts = PromptStore(prompts_path)            # 파일 없으면 즉시 예외
    except (PromptFileMissingError, PromptNotFoundError, PromptConfigError) as e:
        print(f"[FATAL] Prompt config error: {e}")
        return

    emitter_factory = make_console_emitter_factory()

    agentic_cfg = AgenticConfig(cfg_path)
    em_cfg = EmbeddingConfig(cfg_path)
    vs_cfg = VectorStoreConfig(cfg_path)
    retriever_cfg = RetrieverConfig(cfg_path)

    qd  = QdrantSearcher(url=vs_cfg.qdrant_url, collection=vs_cfg.collection)
    enc = HybridEncoder(em_cfg.bgem3, device=DEVICE_EMB)  # 쿼리 dense+sparse 인코딩
    rr  = CrossEncoderReranker(RERANKER_MODEL, device=DEVICE_RERANK)  # 재랭커
    encode_sem = asyncio.Semaphore(1) # bge는 thread non-safe
    rerank_sem = asyncio.Semaphore(1) # bge는 thread non-safe

    policy = LLMPolicy(llms, policy_cfg)

    retriever = HybridRetriever(
        cfg=retriever_cfg,
        qdrant=qd,
        encoder=enc,
        reranker=rr,
        encode_sem=encode_sem,
        rerank_sem=rerank_sem
    )

    # 4) 엔진 생성 (무거운 객체는 __post_init__에서 1회 로드/캐시)
    engine = HybridVectorSearchRAG(
        policy=policy,
        prompt_store=prompts,
        cfg=agentic_cfg,
        emitter_factory=emitter_factory,
        retriever=retriever
    )

    engine.build_graph()


    # 6) 동시 테스트용 질의들
    q1 = "2015년7월1일 불량검증 결과 리포트 형태로 보내줘" # "왜 통행시간 단축이 더 효과적이야?" # "연구에서 적정 요금 할인율 산정 방식과 수치, 최적 근거를 설명해줘"
    q2 = "해당 파일을 기준으로 재 학습요청하고, 도장불량 detect 정확도를 3% 낮춰줘" # "날씨는?"

    # 7) 동시 실행(aretrieve 경로 + 각 노드 스트리밍 확인)
    tasks = [
        engine.run(q1, stream_id="S-001"),
        engine.run(q2, stream_id="S-002"),
    ]
    await asyncio.gather(*tasks)

    # 8) 동일 질의 재실행(캐시/세마포어 체감) — 선택
    #print("\n[repeat] run same query to observe cache reuse / warm paths\n")
    #await engine.run(q1, stream_id="S-003")



if __name__ == "__main__":
    import asyncio
    asyncio.run(async_main())