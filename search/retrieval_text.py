# SPDX-License-Identifier: Apache-2.0

"""
Description: Hybrid dense-sparse retriever with Qdrant, FlagEmbedding (BGEM3), and KeyBERT for keyword extraction.
             Supports PRF query expansion, RRF fusion, cross-encoder re-ranking, adaptive keyword gating, and automatic document locking.
"""



# pip install qdrant-client FlagEmbedding keybert transformers torch sentencepiece tabulate numpy

import asyncio
import pprint
from dataclasses import dataclass, InitVar
from typing import List, Dict, Optional, Tuple, ClassVar
from collections import defaultdict, Counter
import re, math, logging

from qdrant_client import AsyncQdrantClient, models
from FlagEmbedding import BGEM3FlagModel, FlagReranker

from pipelines.hvs_config import VectorStoreConfig, EmbeddingConfig, RetrieverConfig
from search.prf import prf_expand_for_dense_and_sparse

# ===================== 전역 설정 =====================


TEXT_KEYS    = ("page_content",)
DOC_KEY_PATH = ("metadata", "source_file_hash")  # 문서 단위 다양성/락 기준

# 인코딩/재랭커
RERANKER_MODEL     = "BAAI/bge-reranker-v2-m3"
DEVICE_EMB         = "cuda:0"                        # 없으면 "cpu"
DEVICE_RERANK      = "cuda:0"


# ===================== 문서 키워드 추출 (KoBERT+KeyBERT) =====================
HANGUL_RX = re.compile(r"[가-힣]{2,}")
NUM_UNIT_RX = re.compile(r"\d+(?:\.\d+)?%|[₩]|[0-9]+(?:원|만원|억)|\d{4}년|\d+월|\d+일")
STOPWORDS = set([
    "그리고","그러나","또한","이다","있다","수","및","등","부분","대상","결과","통해",
    "기반","사용","본","에서","으로","하는","됐다","한다",
    # ⬇️ 기존 _DEFAULT_STOPWORDS에만 있던 영문 stopword도 포함
    "the","a","an","of","to","in",
])



logger = logging.getLogger("retrieval")
logging.basicConfig(level=logging.INFO)


# ===================== 유틸 =====================
def get_nested(d: dict, path: Tuple[str, ...], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def get_payload_text(payload: dict, keys=TEXT_KEYS) -> str:
    for k in keys:
        v = payload.get(k)
        if v:
            return v
    return ""


# ===================== 인코더/검색기/재랭커 =====================
@dataclass
class HybridEncoder:
    def __init__(self, model: str | BGEM3FlagModel, device: str = DEVICE_EMB):
        if isinstance(model, str):
            self.model = BGEM3FlagModel(model, use_fp16=True, devices=device)
        else:
            self.model = model

    def encode_dense(self, text: str|List[str]) -> Tuple[List[float]] | List[Tuple[List[float]]]:
        embeddings = self.model.encode(text, return_dense=True, return_sparse=False)
        return embeddings["dense_vecs"]

    def encode_both(self, text: str|List[str]) -> Tuple[List[float], Dict[str, List]] | Tuple[List[List[float]], List[Dict[str, List]]]:
        embeddings = self.model.encode(text, return_dense=True, return_sparse=True)
        dense_vecs = embeddings["dense_vecs"]
        sparse_vecs = embeddings["lexical_weights"]

        return dense_vecs, [{
            "indices": list(sv.keys()),
            "values": list(sv.values())
        } for sv in sparse_vecs]


class QdrantSearcher:
    def __init__(self, url: str, collection: str, dense_name: str = "dense", sparse_name: str = "sparse"):
        self.client = AsyncQdrantClient(url=url)
        self.collection = collection
        self.dense_name = dense_name
        self.sparse_name = sparse_name

    async def search_dense(self, vector, limit=10, filter_=None):
        # dense 벡터: list로 전달, 벡터명은 using 파라미터로 지정
        res = await self.client.query_points(
            collection_name=self.collection,
            query=list(map(float, vector)),
            using=self.dense_name,
            query_filter=filter_,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        return list(res.points)

    async def search_sparse(self, indices, values, limit=10, filter_=None):
        # sparse 벡터: SparseVector로 전달, 벡터명은 using 파라미터로 지정
        res = await self.client.query_points(
            collection_name=self.collection,
            query=models.SparseVector(
                indices=list(map(int, indices)),
                values=list(map(float, values)),
            ),
            using=self.sparse_name,
            query_filter=filter_,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return list(res.points)

    async def close(self):
        await self.client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

@dataclass
class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL, device: str = DEVICE_RERANK):
        self.model = FlagReranker(model_name, use_fp16=True, devices=device)

    def rerank(self, query: str, points):
        pairs = [(query, get_payload_text(p.payload)) for p in points]
        scores = self.model.compute_score(pairs, normalize=True)  # 0~1
        reranked = sorted(zip(points, scores), key=lambda x: x[1], reverse=True)
        pts = [p for p,_ in reranked]; scs = [float(s) for _,s in reranked]
        return pts, scs

    # ⬇️ 추가: 비동기 래퍼 (compute_score는 sync이므로 to_thread)
    async def rerank_async(self, query: str, points):
        return await asyncio.to_thread(self.rerank, query, points)


# ===================== 융합/게이팅/락 =====================
def rrf_fuse(dense_points, sparse_points, alpha, kappa):
    score = defaultdict(float)
    for r,p in enumerate(dense_points, 1): score[p.id] += alpha/(kappa+r)
    for r,p in enumerate(sparse_points,1): score[p.id] += (1-alpha)/(kappa+r)
    merged = {p.id:p for p in (dense_points+sparse_points)}
    return sorted(merged.values(), key=lambda p: score[p.id], reverse=True)

def extract_query_terms(q: str, top_k=12):
    toks = [t for t in HANGUL_RX.findall(q) if len(t) >= 2 and t not in STOPWORDS]
    toks += NUM_UNIT_RX.findall(q)
    cnt = Counter(toks)
    return set([t for t,_ in cnt.most_common(top_k)])

def overlap_with_keywords(payload: dict, terms: set[str]) -> float:
    if not terms: return 0.0
    kws = payload.get("keywords") or []
    if kws:
        inter = len(set(kws) & terms)
        return inter / max(1, len(terms))
    # 백업: 본문에서 겹침
    text = get_payload_text(payload)
    words = set([t for t in HANGUL_RX.findall(text) if len(t) >= 2]) | set(NUM_UNIT_RX.findall(text))
    inter = len(terms & words)
    return inter / max(1, len(terms))

def adaptive_keyword_gate_soft(query: str, points, min_overlap, penalty, base_scores=None, hard_cut=False):
    terms = extract_query_terms(query)
    if not terms:
        return points, base_scores
    if base_scores is None:
        base_scores = [1.0]*len(points)

    kept, new_scores = [], []
    kept_any = False
    for p, s in zip(points, base_scores):
        ov = overlap_with_keywords(p.payload or {}, terms)
        if hard_cut and ov < min_overlap:
            continue
        kept_any = kept_any or (ov >= min_overlap)
        adj = s - penalty if ov < min_overlap else s
        kept.append(p); new_scores.append(adj)

    if not kept:
        return points, base_scores  # 안전장치
    ranked = sorted(zip(kept, new_scores), key=lambda x: x[1], reverse=True)
    pts = [p for p,_ in ranked]; scs = [s for _,s in ranked]
    return pts, scs


# ===================== PRF(의사 관련 피드백) & sparse 폴백 =====================
def _sparse_ok(svec: dict) -> bool:
    try:
        return bool(svec and svec.get("indices") and len(svec["indices"]) > 0)
    except Exception:
        return False


# ===================== 리트리버 =====================
@dataclass
class HybridRetriever:
    cfg: InitVar[RetrieverConfig]
    qdrant: QdrantSearcher
    encoder: HybridEncoder
    reranker: Optional[CrossEncoderReranker] = None
    use_keyword_gate: bool = True
    use_rerank: bool = True
    enable_prf_expand: bool = True
    prf_top_m: int = 12
    prf_max_terms_dense: int = 3
    prf_max_terms_sparse: int = 5
    # ⬇️ 추가: 리랭크 동시성 제한
    rerank_sem: Optional[asyncio.Semaphore] = None  # 교차인코더 동시성 제한(옵션)
    encode_sem: Optional[asyncio.Semaphore] = None  # (선택) 인코딩 동시성 제한

    CFG: ClassVar[RetrieverConfig] = None


    def __post_init__(self, cfg: RetrieverConfig):
        # 초기화 시점에 설정값을 CFG로 저장
        self.configure(cfg)

    @classmethod
    def configure(cls, cfg: RetrieverConfig):
        cls.CFG = cfg


    async def aretrieve(self, query: str, filter_: Optional[models.Filter] = None):
        return await self.aretrieve_many([query], filter_=filter_)


    async def _aretrieve(self, query: str, filter_: Optional[models.Filter] = None):
        """
        사용자 질문인 query를 dense/sparse로 임베딩함
        sparse는 가끔 뻑나니깐, if _sparse_ok(svec)로 체크해서 재검색 시도. 그래도 안되면 dense만 쓰는거임
        여튼 query의 임베딩값으로 dense/sparse를 검색해서 리랭킹함
        """
        if self.encode_sem:
            async with self.encode_sem:
                dvec, svec = await asyncio.to_thread(self.encoder.encode_both, query)
                dense = await self.qdrant.search_dense(dvec, self.CFG.dense_k, filter_)
        else:
            dvec, svec = await asyncio.to_thread(self.encoder.encode_both, query)
            dense = await self.qdrant.search_dense(dvec, self.CFG.dense_k, filter_)

        # 3) sparse 검색 or PRF 폴백
        sparse_points = []
        used_sparse = False
        if _sparse_ok(svec):
            sparse_points = await self.qdrant.search_sparse(svec["indices"], svec["values"], self.CFG.sparse_k, filter_)
            used_sparse = True
            print("=== Sparse 벡터 검색 성공 ===")
        else:
            logger.warning("[retrieval] sparse unavailable -> fallback to dense-only.")
            if self.enable_prf_expand and dense:
                prf = prf_expand_for_dense_and_sparse(query, dense, stopwords=STOPWORDS, top_m=self.prf_top_m, max_terms_dense=self.prf_max_terms_dense, max_terms_sparse=self.prf_max_terms_sparse)
                print("=== PRF 확장 쿼리 ===")
                pprint.pprint(prf["dense_query"])
                dvec2 = await asyncio.to_thread(self.encoder.encode_dense, prf["dense_query"])
                dense = await self.qdrant.search_dense(dvec2, self.CFG.dense_k, filter_)

        print("=== 오리지널 디비검색 결과 ===")
        print("dense vectors:")
        pprint.pprint(dense)
        print("sparse vectors:")
        pprint.pprint(sparse_points)

        # 4) 융합
        fused = rrf_fuse(dense, sparse_points, alpha=self.CFG.alpha, kappa=self.CFG.kappa) if used_sparse else dense
        print("=== RRF FUSE 결과 ===")
        pprint.pprint(fused)

        # 5) 재랭크 (세마포어로 보호)
        candidates = fused[:max(self.CFG.final_k, 50)]
        scores = None
        if self.use_rerank and self.reranker is not None and candidates:
            async def _do():
                return await self.reranker.rerank_async(query, candidates)

            if self.rerank_sem:
                async with self.rerank_sem:
                    candidates, scores = await _do()
            else:
                candidates, scores = await _do()

        print("=== reranker 결과 ===")
        pprint.pprint(candidates)

        # 5.5) 키워드 게이팅
        if self.use_keyword_gate:
            candidates, scores = adaptive_keyword_gate_soft(
                query, candidates,
                min_overlap=self.CFG.min_overlap, penalty=self.CFG.penalty, base_scores=scores, hard_cut=False
            )

        print("=== 키워드 게이팅 결과 ===")
        pprint.pprint(candidates)

        # 6) 최소 점수 컷
        candidates, scores = self._apply_min_score_cut(candidates, scores, min_score=self.CFG.min_score)
        print("=== 최소 점수 컷 결과 ===")
        pprint.pprint(candidates)

        # 7) 자동 문서 락 + per-doc cap
        final = self.finalize_with_autolock(
            candidates, scores=scores,
            doc_id_getter=lambda p: get_nested(p.payload or {}, DOC_KEY_PATH)
        )
        print("=== 자동 문서 락 결과 ===")
        pprint.pprint(final)

        # rerank 점수 부착
        try:
            if scores is not None:
                idx_map = {id(c): i for i, c in enumerate(candidates)}
                for p in final:
                    i = idx_map.get(id(p))
                    if i is not None and i < len(scores):
                        p.score = float(scores[i])
        except Exception:
            pass
        return final

    async def _search_one(self, query: str, dense, sparse_points, used_sparse: bool):
        # 4) 융합
        fused = rrf_fuse(dense, sparse_points, alpha=self.CFG.alpha, kappa=self.CFG.kappa) if used_sparse else dense
        print("=== RRF FUSE 결과 ===")
        pprint.pprint(fused)

        # 5) 재랭크 (세마포어로 보호)
        candidates = fused[:max(self.CFG.final_k, 50)]
        scores = None
        if self.use_rerank and self.reranker is not None and candidates:
            print("=== 재랭크 대상 ===")
            async def _do():
                return await self.reranker.rerank_async(query, candidates)

            if self.rerank_sem:
                async with self.rerank_sem:
                    candidates, scores = await _do()
            else:
                candidates, scores = await _do()

        print("=== reranker 결과 ===")
        pprint.pprint(candidates)

        # 5.5) 키워드 게이팅
        if self.use_keyword_gate:
            candidates, scores = adaptive_keyword_gate_soft(
                query, candidates,
                min_overlap=self.CFG.min_overlap, penalty=self.CFG.penalty, base_scores=scores, hard_cut=False
            )

        print("=== 키워드 게이팅 결과 ===")
        pprint.pprint(candidates)

        # 6) 최소 점수 컷
        candidates, scores = self._apply_min_score_cut(candidates, scores, min_score=self.CFG.min_score)
        print("=== 최소 점수 컷 결과 ===")
        pprint.pprint(candidates)

        # 7) 자동 문서 락 + per-doc cap
        final = self.finalize_with_autolock(
            candidates, scores=scores,
            doc_id_getter=lambda p: get_nested(p.payload or {}, DOC_KEY_PATH)
        )
        print("=== 자동 문서 락 결과 ===")
        pprint.pprint(final)

        # rerank 점수 부착
        try:
            if scores is not None:
                idx_map = {id(c): i for i, c in enumerate(candidates)}
                for p in final:
                    i = idx_map.get(id(p))
                    if i is not None and i < len(scores):
                        p.score = float(scores[i])
        except Exception:
            pass
        return final


    async def aretrieve_many(self, queries: list[str], filter_: Optional[models.Filter]=None, batch_size: int = 64):
        if not queries:
            return []

        # 1) dense+sparse 1차 배치 인코딩
        dvecs, svecs = await asyncio.to_thread(self.encoder.encode_both, queries)

        # 2) sparse 없는 애들만 확장 → dense 2차 배치 인코딩 (필요한 부분만)
        to_fix_idx = [i for i, s in enumerate(svecs) if not s or not s.get("indices")]
        if not to_fix_idx:
            async def _sparse(i: int):
                return await self.qdrant.search_sparse(svecs[i]["indices"], svecs[i]["values"], self.CFG.sparse_k, filter_)
            sparse_points = await asyncio.gather(*[_sparse(i) for i in range(len(queries))])
            used_sparse = True
        else:
            sparse_points = []
            used_sparse = False
            expanded_texts = [prf_expand_for_dense_and_sparse(queries[i], dvecs[i],
                                                  stopwords=STOPWORDS, top_m=self.prf_top_m,
                                                  max_terms_dense=self.prf_max_terms_dense,
                                                  max_terms_sparse=self.prf_max_terms_sparse)
            for i in to_fix_idx]
            expanded_texts = [et["dense_query"] for et in expanded_texts]
            dense_fixed = await asyncio.to_thread(self.encoder.encode_dense, expanded_texts)
            # 해당 인덱스의 dense 벡터를 교체
            for j, i in enumerate(to_fix_idx):
                dvecs[i] = dense_fixed[j]

        async def _dense(i: int):
            return await self.qdrant.search_dense(dvecs[i], self.CFG.dense_k, filter_)
        dense_points = await asyncio.gather(*[_dense(i) for i in range(len(queries))])

        # 3) 검색은 쿼리별 병렬
        async def _one(i: int):
            return await self._search_one(
                query=queries[i],
                dense=dense_points[i],
                sparse_points=sparse_points[i] if used_sparse else [],
                used_sparse=used_sparse
            )

        tasks = [_one(i) for i in range(len(queries))]
        return await asyncio.gather(*tasks, return_exceptions=False)


    @staticmethod
    def _apply_min_score_cut(candidates, scores, min_score: float):
        if scores is None:  # 재랭크 안 쓴 경우 패스
            return candidates, scores
        kept = [(p, s) for p, s in zip(candidates, scores) if s >= min_score]
        if not kept:
            return candidates, scores  # 전부 날아가면 안전장치로 원본 유지
        kept.sort(key=lambda x: x[1], reverse=True)
        pts = [p for p, _ in kept]
        scs = [float(s) for _, s in kept]
        return pts, scs

    @classmethod
    def auto_doc_lock(cls,
                      points, scores=None,
                      doc_id_getter=lambda p: get_nested(p.payload or {}, DOC_KEY_PATH)):
        lookahead = cls.CFG.auto_lock_lookahead
        share_threshold = cls.CFG.auto_lock_share_thresh
        margin_threshold = cls.CFG.auto_lock_margin_thresh

        cand = points[:lookahead]
        if not cand: return None
        docs = [doc_id_getter(p) for p in cand]
        cnt = Counter(docs)
        top_doc, top_cnt = cnt.most_common(1)[0]
        share = top_cnt / max(1, len(cand))
        if share < share_threshold:
            return None
        if scores is not None:
            by_doc = defaultdict(list)
            for p, s in zip(cand, scores[:len(cand)]):
                by_doc[doc_id_getter(p)].append(float(s))
            means = sorted(((d, sum(v) / len(v)) for d, v in by_doc.items()), key=lambda x: x[1], reverse=True)
            margin = means[0][1] - (means[1][1] if len(means) >= 2 else 0.0)
            if margin < margin_threshold:
                return None
        return top_doc

    @classmethod
    def finalize_with_autolock(cls,
                               candidates,
                               scores=None,
                               doc_id_getter=lambda p: get_nested(p.payload or {}, DOC_KEY_PATH) ):
        final_k = cls.CFG.final_k
        per_doc_cap = cls.CFG.per_doc_cap

        lock_doc = cls.auto_doc_lock(candidates, scores=scores, doc_id_getter=doc_id_getter)
        if lock_doc is not None:
            locked = [p for p in candidates if doc_id_getter(p) == lock_doc]
            return locked[:final_k]
        used, out = Counter(), []
        for p in candidates:
            d = doc_id_getter(p)
            if used[d] >= per_doc_cap:
                continue
            used[d] += 1
            out.append(p)
            if len(out) >= final_k:
                break
        return out


# ===================== 간단 지표(선택) =====================
def precision_at_k(ids, rel, k):
    s=set(rel); top=ids[:k]
    return (sum(1 for x in top if x in s)/len(top)) if top else 0.0
def recall_at_k(ids, rel, k):
    s=set(rel); top=ids[:k]
    return (sum(1 for x in top if x in s)/len(s)) if s else 0.0
def dcg_at_k(ids, rel, k):
    s=set(rel); dcg=0.0
    for i,pid in enumerate(ids[:k],1): dcg += (1.0 if pid in s else 0.0)/math.log2(i+1)
    return dcg
def ndcg_at_k(ids, rel, k):
    ideal=dcg_at_k(list(rel)[:k], rel, k)
    return (dcg_at_k(ids, rel, k)/ideal) if ideal>0 else 0.0



async def main():
    cfg_path = "../config/rag.yaml"
    vs_cfg = VectorStoreConfig(cfg_path)
    em_cfg = EmbeddingConfig(cfg_path)

    # 1) 검색기/인코더/재랭커
    qd  = QdrantSearcher(url=vs_cfg.qdrant_url, collection=vs_cfg.collection)
    enc = HybridEncoder(em_cfg.bgem3, device=DEVICE_EMB)  # 쿼리 dense+sparse 인코딩
    rr  = CrossEncoderReranker(RERANKER_MODEL, device=DEVICE_RERANK)  # 재랭커

    # (선택) 동시성 제한: 인코딩/리랭크가 무거우면 세마포어 넣어도 좋음
    encode_sem = asyncio.Semaphore(2) # bge는 thread non-safe
    rerank_sem = asyncio.Semaphore(2) # bge는 thread non-safe

    retriever = HybridRetriever(
        cfg=RetrieverConfig(cfg_path),
        qdrant=qd,
        encoder=enc,
        reranker=rr,
        encode_sem=encode_sem, # 인코딩 동시성 제한. 안넣으면 제한 안함
        rerank_sem=rerank_sem, # 리랭크 동시성 제한. 안넣으면 제한 안함
    )

    print("HybridRetriever initialized.")
    query = "왜 통행시간 단축이 더 효과적이야?"
    #"통행시간 단축이 교통 혼잡 완화에 더 효과적인 이유는 무엇인가요?"

            #'연구에서 “적정 요금 할인율”을 산정한 방식은 무엇이며, 그 수치는 얼마였고 왜 그 수준이 최적이라고 판단되었을까요?'
    results = await retriever.aretrieve(query)
    results = results[0]
    print(f"Retrieved {len(results)} documents for query: {query}")
    for i, p in enumerate(results, 1):
        doc_key = get_nested(p.payload or {}, DOC_KEY_PATH)
        txt = get_payload_text(p.payload).replace("\n"," ")[:180]
        score = getattr(p, "score", None)  # aretrieve가 최종 점수를 p.score에 넣어둠
        if score is not None:
            print(f"{i:2d}. id={p.id}  doc={doc_key}  score={score:.3f}  text={txt}…")
        else:
            print(f"{i:2d}. id={p.id}  doc={doc_key}  text={txt}…")


    # 병렬 테스트
    # queries = ["대중교통", "적정 요금 할인율", "시간 단축"]
    # batches = await asyncio.gather(*[retriever.aretrieve(q) for q in queries])
    # print(f"Retrieved {len(batches)} documents for queries: {queries}")
    # import pprint
    # pprint.pprint(batches)



# ===================== 실행 예시 =====================
if __name__ == "__main__":
    asyncio.run(main())
