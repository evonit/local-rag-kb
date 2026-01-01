# SPDX-License-Identifier: Apache-2.0

"""
Description: API Severizing module for RAG (Retrieval-Augmented Generation) with WebSocket support.
"""



import asyncio
import contextlib
import json
import uuid
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from kiwipiepy import Kiwi

from pipelines.hvs_config import AgenticConfig, EmbeddingConfig, VectorStoreConfig, RetrieverConfig
from pipelines.hvs_rag import HybridVectorSearchRAG
from search.keyword_extractor import extract_keywords_embedrank
from dataloader.loader import run_async, get_subject_from_llm_async
from models.llm_model import load_llms
from models.llm_policy import load_policy_cfg, LLMPolicy
from prompts.prompt_store import PromptStore, PromptFileMissingError, PromptNotFoundError, PromptConfigError
from search.retrieval_text import QdrantSearcher, HybridEncoder, DEVICE_EMB, DEVICE_RERANK, RERANKER_MODEL, \
    CrossEncoderReranker, HybridRetriever
from shared_types.emitters.queueing_emitter import make_queueing_emitter_factory

import logging
logger = logging.getLogger("uvicorn")

from concurrent.futures import ThreadPoolExecutor
loop = asyncio.get_running_loop()
loop.set_default_executor(ThreadPoolExecutor(max_workers=8))

try:
    llms = load_llms()  # small/large 모델 번들
    policy_cfg = load_policy_cfg()  # LLMPolicyConfig
    prompts = PromptStore()  # 파일 없으면 즉시 예외
except (PromptFileMissingError, PromptNotFoundError, PromptConfigError) as e:
    print(f"[FATAL] Prompt config error: {e}")
    exit(1)


agentic_cfg = AgenticConfig()
em_cfg = EmbeddingConfig()
vs_cfg = VectorStoreConfig()
retriever_cfg = RetrieverConfig()

print("agentic_cfg:", agentic_cfg.__dict__)
print("em_cfg:", em_cfg.__dict__)
print("vs_cfg:", vs_cfg.__dict__)
print("retriever_cfg:", retriever_cfg.__dict__)

qd  = QdrantSearcher(url=vs_cfg.qdrant_url, collection=vs_cfg.collection)
enc = HybridEncoder(em_cfg.bgem3, device=DEVICE_EMB)  # 쿼리 dense+sparse 인코딩
rr  = CrossEncoderReranker(RERANKER_MODEL, device=DEVICE_RERANK)  # 재랭커
encode_sem = asyncio.Semaphore(1) # bge는 thread non-safe
rerank_sem = asyncio.Semaphore(1) # bge는 thread non-safe

policy = LLMPolicy(llms, policy_cfg)

kiwi = Kiwi(num_workers=4)

retriever = HybridRetriever(
    cfg=retriever_cfg,
    qdrant=qd,
    encoder=enc,
    reranker=rr,
    encode_sem=encode_sem,
    rerank_sem=rerank_sem
)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")



@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    emitter_factory, writer_task = make_queueing_emitter_factory(ws)
    workflow = HybridVectorSearchRAG(
        policy=policy,
        prompt_store=prompts,
        cfg=agentic_cfg,
        emitter_factory=emitter_factory,
        retriever=retriever
    )

    workflow.build_graph()

    tasks: set[asyncio.Task] = set()
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "query.start":
                stream_id = msg.get("stream_id") or str(uuid.uuid4())
                query = msg["query"].strip()
                if msg.get("options"):
                    #options = QueryOptions(**(msg.get("options")))
                    #print("query options: ", options)
                    options = None
                else:
                    options = None

                async def run_query():
                    try:
                        await workflow.run(query=query, stream_id=stream_id, options=options)
                    except asyncio.CancelledError:
                        # 취소된 경우: 선택적으로 알림
                        em = emitter_factory(stream_id)
                        await em.event(node="workflow", stream_id=stream_id, event="cancelled")
                        raise
                    except Exception as e:  # noqa: BLE001
                        logger.error(e, exc_info=True)
                        em = emitter_factory(stream_id)
                        await em.json(
                            node="workflow",
                            stream_id=stream_id,
                            payload={"error": str(e)},
                        )
                        await em.done(node="workflow", stream_id=stream_id)

                task = asyncio.create_task(run_query())
                tasks.add(task)
                task.add_done_callback(lambda t: tasks.discard(t))

            elif mtype == "control":
                action = msg.get("action")
                # 여기서는 간단히 모든 실행을 취소하는 예시. 실제로는 stream_id별 관리 필요.
                if action == "pause":
                    for t in list(tasks):
                        t.cancel()
                elif action == "resume":
                    # 재개는 보통 "from_stream"의 저장 상태로 새 run() 호출
                    pass
                elif action == "switch_model":
                    # 현재 작업을 중단하고 새 모델로 run() 재시작
                    pass
            else:
                em = emitter_factory("-")
                await em.json(node="server", stream_id="-", payload={"warn": "unsupported message", "raw": msg})

    except WebSocketDisconnect:
        pass
    finally:
        # 실행 중 태스크 정리
        for t in list(tasks):
            t.cancel()
        # writer 종료
        writer_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await writer_task


@app.get("/api/reload_prompts")
async def reload_prompts():
    prompts.reload()
    return {"status": "prompts reloaded"}


# upload pdf 파일 and run vectorization
upload_lock = asyncio.Lock()
@app.post("/api/upload_and_embbeding_pdf")
async def upload_file(files: List[UploadFile] = File(...)):
    async with upload_lock:
        get_subject_from_llm_func = lambda text: get_subject_from_llm_async(llms.small, text)
        keyword_extract_func = lambda text: extract_keywords_embedrank(em=enc.model, kiwi=kiwi, text=text, top_n=10,
                                                                       diversity=0.6, ngram_range=(1, 2))

        for file in files:
            if file.filename.endswith(".pdf"):
                logger.info(f"Processing file: {file.filename}")
                await run_async(vs_cfg.collection, file, enc.model, em_cfg.tokenizer, qd.client, get_subject_from_llm_func, keyword_extract_func)

    return {"status": "pdf is reloaded into vector store"}
