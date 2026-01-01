# SPDX-License-Identifier: Apache-2.0

"""
Description: Extract keywords from text using a pre-trained tokenizer.
"""

# 설치 (필요 시)
# pip install "FlagEmbedding>=1.2.10" kiwipiepy numpy torch

from typing import List, Tuple, Iterable
import numpy as np

# 1) 임베딩 모델: KURE-v1 (BGEM3 계열)
from FlagEmbedding import BGEM3FlagModel

# 2) 형태소 분석기: Kiwi (빠르고 설치 간단)
from kiwipiepy import Kiwi

# 3) 한국어 불용어(가벼운 기본셋; 도메인에 맞게 확장 권장)
KO_STOPWORDS = set("""
하다 되다 이다 있다 없다 그리고 그러나 또한 즉 및 등의 등 등등 보다 통해 대한 대해서 위해 위하여 같이 같은 경우 때 때는 등이 로서 로써 으로 으로서 에서 에에 에게 한테 그 그에 그에대한 그가 그녀 그들 이 그건 그럼 그런 이런 저런 아주 보다 매우 너무 다소 약간 더욱 가장 더 많이 적게 등등 요 등등요
""".split())

# ===== 형태소 기반 전처리 =====

def _strip_particles_tokens(tokens: Iterable) -> List[str]:
    """
    Kiwi 토큰 시퀀스에서 조사/어미/접사 제거하고, 불필요한 기호/한글자 제거.
    - J*: 조사, E*: 어미, X*: 접사(파생/어근결합) 제거
    """
    kept = []
    for t in tokens:
        tag = t.tag  # JKS, JKO, JKB, EF, EC, ETN, ETM, XSN, XSV, ...
        if tag.startswith('J') or tag.startswith('E') or tag.startswith('X'):
            continue
        form = t.form.strip()
        if not form:
            continue
        # 너무 짧은 토큰(한 글자 숫자/기호 등) 제거(도메인에 따라 조정)
        if len(form) == 1 and not form.isdigit():
            continue
        kept.append(form)
    return kept

def normalize_text_for_candidates(kiwi: Kiwi, text: str) -> List[str]:
    """
    텍스트 → Kiwi 품사분석 → 조사/어미/접사 제거 → 토큰 리스트
    불용어 간단 제거(도메인 확장 권장)
    """
    tokens = kiwi.tokenize(text)
    kept = _strip_particles_tokens(tokens)
    kept2 = [w for w in kept if w.lower() not in KO_STOPWORDS]
    return kept2

# ===== 후보 생성 (n-gram) =====

def generate_ngram_candidates(tokens: List[str], ngram_range=(1, 2),
                              min_char=2, max_candidates=2000) -> List[str]:
    """
    형태소 정제된 토큰들로부터 n-gram 후보 생성.
    - min_char: 너무 짧은 후보 제외
    - max_candidates: 방어적 상한
    """
    cands = []
    n_min, n_max = ngram_range
    for n in range(n_min, n_max + 1):
        for i in range(0, len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i+n]).strip()
            if len(phrase.replace(" ", "")) < min_char:
                continue
            cands.append(phrase)
            if len(cands) >= max_candidates:
                break
        if len(cands) >= max_candidates:
            break
    # 중복 제거(순서 유지)
    seen = set()
    uniq = []
    for p in cands:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq

# ===== 임베딩 & 점수 =====

def _embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    BGEM3(KURE-v1) 임베딩. dense 벡터만 사용.
    """
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        out = _embedder.encode(batch, batch_size=len(batch))  # returns dict
        # BGEM3FlagModel.encode -> {'dense_vecs': np.ndarray, 'sparse_vecs': ...}
        dense = out["dense_vecs"]
        vecs.append(dense)
    return np.vstack(vecs)


def embed_texts_all_at_once(em: BGEM3FlagModel, texts: List[str]) -> np.ndarray:
    """
    BGEM3(KURE-v1) 임베딩 - 전체 배열을 한 번에 encode 호출.
    반환: (N, dim) numpy 배열
    """
    if not texts:
        return np.empty((0, em.model.config.hidden_size))

    out = em.encode(texts, batch_size=len(texts))  # 한 번에 처리
    return out["dense_vecs"]  # (N, dim)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (d,), b: (N, d) -> (N,)
    """
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return b_norm @ a_norm

# ===== MMR(Optional) 재순위 =====

def mmr_rerank(doc_vec: np.ndarray, cand_vecs: np.ndarray, candidates: List[str],
               top_n: int = 8, diversity: float = 0.5) -> List[Tuple[str, float]]:
    """
    Maximal Marginal Relevance.
    diversity=1.0이면 다양성 극대화, 0이면 유사도만.
    """
    sims = _cosine_sim(doc_vec, cand_vecs)  # (N,)
    selected, selected_idx = [], []
    cand_idx = list(range(len(candidates)))

    # 첫 번째는 최대 유사도
    first = int(np.argmax(sims))
    selected.append((candidates[first], float(sims[first])))
    selected_idx.append(first)
    cand_idx.remove(first)

    while len(selected) < min(top_n, len(candidates)) and cand_idx:
        mmr_scores = []
        for j in cand_idx:
            # diversity * (1 - max sim to selected) + (1-diversity) * sim to doc
            if selected_idx:
                sim_to_selected = np.max(cand_vecs[j] @ (cand_vecs[selected_idx].T) /
                                         ((np.linalg.norm(cand_vecs[j]) + 1e-9) *
                                          (np.linalg.norm(cand_vecs[selected_idx], axis=1) + 1e-9)))
            else:
                sim_to_selected = 0.0
            score = (1 - diversity) * sims[j] - diversity * sim_to_selected
            mmr_scores.append((j, score))
        j_star = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append((candidates[j_star], float(sims[j_star])))
        selected_idx.append(j_star)
        cand_idx.remove(j_star)

    return selected

# ===== 메인 함수 =====

def extract_keywords_embedrank(
    em: BGEM3FlagModel,
    kiwi: Kiwi,
    text: str,
    top_n: int = 8,
    diversity: float = 0.5,
    ngram_range=(1, 2),
    min_char: int = 2
) -> List[str]:
    """
    KURE-v1(BGEM3) + EmbedRank + Kiwi 후처리 파이프라인.
    반환: 정제된 키워드 리스트(점수는 내부에서 사용).
    """
    if not text or not text.strip():
        return []

    # 1) 형태소 기반 정제 토큰
    tokens = normalize_text_for_candidates(kiwi, text)

    # 2) n-gram 후보 생성
    candidates = generate_ngram_candidates(tokens, ngram_range=ngram_range, min_char=min_char)
    if not candidates:
        return []

    # 3) 문서 + 후보 임베딩 (배치)
    doc_vec = embed_texts_all_at_once(em, [text])[0]
    cand_vecs = embed_texts_all_at_once(em, candidates)

    # 4) MMR 재순위로 다양성 확보
    ranked = mmr_rerank(doc_vec, cand_vecs, candidates, top_n=top_n, diversity=diversity)

    # 5) 최종 키워드만 반환(중복/짧은 토큰은 앞에서 이미 정리)
    final_keywords = [k for k, _ in ranked]
    return final_keywords

# ===== 사용 예시 =====
if __name__ == "__main__":
    EMB_MODEL_NAME = "nlpai-lab/kure-v1"
    _embedder = BGEM3FlagModel(EMB_MODEL_NAME, use_fp16=True)  # GPU 있으면 자동사용
    _kiwi = Kiwi(num_workers=4)

    sample = "자율주행자동차의 시뮬레이션에서 v_set_cc 값에 따른 의도치 않은 가속이 발생했습니다. 통행시간 단축과 요금 감면의 효과를 비교했습니다."
    for _ in range(10):
        print(extract_keywords_embedrank(em=_embedder, kiwi=_kiwi, text=sample, top_n=10, diversity=0.6, ngram_range=(1,2)))
