# SPDX-License-Identifier: Apache-2.0

"""
Description: Pseudo-Relevance Feedback (PRF) for query expansion in information retrieval.
             This module provides functionality to expand search queries using top-ranked documents
             by extracting and re-weighting keywords and phrases.
"""



from typing import List, Mapping, Any, Iterable, Tuple, Optional
from collections import defaultdict
import re

_PHRASE_RX  = re.compile(r"[가-힣A-Za-z0-9%](?:[ 가-힣A-Za-z0-9%]*[가-힣A-Za-z0-9%])?")
_DEFAULT_STOPWORDS = {
    "그리고","그러나","또한","이다","있다","수","및","등","부분","대상","결과","통해",
    "기반","사용","본","에서","으로","하는","됐다","한다","the","a","an","of","to","in",
}


def _norm(s: str) -> str:
    return s.lower().strip()

def _get_nested(payload: dict, dotted: str):
    cur = payload
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur

def _iter_keywords(obj: Any) -> Iterable[Tuple[str, float]]:
    if obj is None:
        return
    if isinstance(obj, dict):
        for t, w in obj.items():
            if isinstance(t, str):
                yield (_norm(t), float(w) if isinstance(w, (int, float)) else 1.0)
        return
    if isinstance(obj, (list, tuple)):
        for it in obj:
            if isinstance(it, str):
                yield (_norm(it), 1.0)
            elif isinstance(it, (list, tuple)) and len(it) >= 1:
                t = _norm(str(it[0]))
                w = float(it[1]) if len(it) >= 2 and isinstance(it[1], (int, float)) else 1.0
                yield (t, w)
            elif isinstance(it, dict):
                term = it.get("term") or it.get("text") or it.get("keyword") or it.get("key")
                if term is None:
                    continue
                weight = it.get("weight", it.get("score", 1.0))
                yield (_norm(str(term)), float(weight) if isinstance(weight, (int, float)) else 1.0)

def prf_expand_for_dense_and_sparse(
    query: str,
    candidates: List[Any],                 # list[ScoredPoint]
    top_m: int = 12,
    *,
    keywords_field: str = "metadata.keywords",  # 너의 데이터에 맞춤
    idf: Optional[Mapping[str, float]] = None,
    stopwords: Optional[set[str]] = None,
    per_doc_cap: int = 10,
    rrf_k: int = 60,
    min_len: int = 2,
    # 출력 제어
    max_terms_sparse: int = 8,            # sparse는 조금 더 넉넉히
    max_terms_dense: int  = 4,            # dense는 더 타이트하게(드리프트 억제)
    quote_phrases_sparse: bool = True,    # sparse는 "..." 권장
) -> dict:
    """
    상위 문서의 phrase/단어 키워드를 문서 랭크(RRF) × 키워드 가중 × (선택적 IDF)로 재가중.
    → sparse용(따옴표)과 dense용(무따옴표) 확장 문자열을 각각 생성해 반환.
    반환: {
      "sparse_query": "<원쿼리> \"키워드A B\" \"키워드C\" ...",
      "dense_query":  "<원쿼리> 키워드A B 키워드C ...",
      "added_terms_sparse": [...],
      "added_terms_dense": [...]
    }
    """
    if len(candidates) == 0:
        return {
            "sparse_query": query,
            "dense_query": query,
            "added_terms_sparse": [],
            "added_terms_dense": [],
        }

    stop = stopwords if stopwords is not None else _DEFAULT_STOPWORDS

    def _valid_phrase(p: str) -> bool:
        p = p.strip()
        if len(p) < min_len:
            return False
        parts = [x for x in p.split() if len(x) >= min_len]
        if not parts or all((x in stop) for x in parts):
            return False
        return _PHRASE_RX.fullmatch(p) is not None

    # 1) 후보 가중치 집계
    term_weight: dict[str, float] = defaultdict(float)
    top = candidates[: min(top_m, len(candidates))]

    q_low = query.lower()

    for rank, sp in enumerate(top):
        payload = getattr(sp, "payload", {}) or {}
        kws_raw = _get_nested(payload, keywords_field)
        if kws_raw is None:
            kws_raw = payload.get("keywords") or payload.get("metadata", {}).get("keywords")
        if not kws_raw:
            continue

        w_doc = 1.0 / (rrf_k + rank)  # RRF rank weight
        added = 0

        for raw_term, w_kw in _iter_keywords(kws_raw):
            # phrase를 항상 한 단위로 사용
            if not _valid_phrase(raw_term):
                continue
            # 원 쿼리에 이미 있는 경우는 제외(부분 포함까지 대략 제한)
            if raw_term in q_low:
                continue

            w = w_doc * (w_kw if (isinstance(w_kw,(int,float)) and w_kw > 0) else 1.0)
            if idf is not None:
                toks = [t for t in raw_term.split() if len(t) >= min_len]
                if toks:
                    w *= sum(float(idf.get(t, 1.0)) for t in toks) / len(toks)

            term_weight[raw_term] += w
            added += 1
            if added >= per_doc_cap:
                break

    if not term_weight:
        return {
            "sparse_query": query,
            "dense_query": query,
            "added_terms_sparse": [],
            "added_terms_dense": [],
        }

    ranked = sorted(term_weight.items(), key=lambda kv: kv[1], reverse=True)

    # 2) sparse용 선택(따옴표)
    added_sparse: list[str] = []
    for term, _ in ranked:
        if len(added_sparse) >= max_terms_sparse:
            break
        added_sparse.append(f"\"{term}\"" if quote_phrases_sparse else term)

    # 3) dense용 선택(무따옴표, 더 타이트)
    added_dense: list[str] = []
    for term, _ in ranked:
        if len(added_dense) >= max_terms_dense:
            break
        added_dense.append(term)

    sparse_query = (query + " " + " ".join(added_sparse)).strip() if added_sparse else query
    dense_query  = (query + " " + " ".join(added_dense)).strip()  if added_dense  else query

    return {
        "sparse_query": sparse_query,
        "dense_query": dense_query,
        "added_terms_sparse": added_sparse,
        "added_terms_dense": added_dense,
    }



if __name__ == "__main__":
    # candidates: list[ScoredPoint], payload에 {"keywords": [...]} 가 있다고 가정
    from qdrant_client.http.models import ScoredPoint
    candidates = [
        ScoredPoint(id='3cc394d5-628a-13b9-42cc-8774ea535691', version=21, score=0.9706321458187713, payload={'page_content': 'Alt 2 44.7% 10.6% 24.3% 13.2% 7.2% -2.7% 0.4% 1.5% 0.6% 0.1% 112.5\n4. 통행시간 및 비용 감소에 따른 변화(시나리오3) 및 적정할인률 산정\n마지막으로 시나리오 3에서는 MaaS를 통해 통행시간과 통행비용이 동시에 감소할 때 수익금의 손실이 발\n생하지 않는 적정 할인율을 분석하였다. 시나리오1의 분석 결과 요금할인율을 10% 적용했을 때 3.73억원의\n10 한국ITS학회논문지 제18권, 제1호(2019년 2월)\nMasS(Mobility as a Service)의 적정요금할인 수준 분석\n수입금이 감소하는 것으로 나타났기 때문에 시나리오3에서는 손실이 최소화되도록 할인율을 5% 이내로 한\n정하였다. 통행시간의 감소는 시나리오2의 2안인 택시와 대중교통의 통행시간 감소를 모두 고려한 상태를\n기준으로 하였다. 분석결과 시나리오1과 같이 할인율이 증가할수록 대중교통 수요는 증가하였으나 수입금은\n요금할인율이 3%에 이르면 감소하기 시작하는 것으로 나타났다(Table 9).\n<Table 9> Analysis Result of Scenario 3 (Travel Cost & Time Reduction)\nModal split ratio(%) Modal split change(%)\nDiscount Revenue\nBus/ Bus/\nrate(%) Car Taxi Bus Subway Car Taxi Bus Subway (100 million won)\nSubway Subway\n0% 44.7% 10.6% 24.3% 13.2% 7.2% -2.7% 0.4% 1.5% 0.6% 0.1% 112.5\n1% 44.7% 10.6% 24.3% 13.2% 7.2% -2.7% 0.4% 1.5% 0.6% 0.1% 73.1\n2% 44.7% 10.6% 24.3% 13.2% 7.2% -2.7% 0.4% 1.5% 0.6% 0.1% 35.7', 'metadata': {'source_id': '3cc394d5-628a-13b9-42cc-8774ea535691', 'source_file_hash': '48b23d003a93fb3381fd982b6afd61f588519932a8b861dfa89a5d9750310de6', 'source_file_name': '/home/knpu/다운로드/교통 관련 10년치 학회지_언어모델 생성용/ITS/ITS/2019/VOL 18. N 1/00018_001_1.pdf', 'source_subject': 'MaaS 도입의 적정 요금 할인율 분석', 'pages': [10, 11], 'keywords': ['감소 시나리오', '통행 비용', 'MaaS', 'Modal split', '3 적정', '학회 논문', '한국 ITS', '분석 수입금', 'Subway 0', 'Mobility as']}}, vector=None, shard_key=None, order_value=None),
        ScoredPoint(id='57845459-7165-710b-4e47-014e0832b329', version=21, score=0.9522199336364293, payload={'page_content': '한국ITS학회논문지 J. Korea Inst. Intell. Transp. Syst.\nMasS(Mobility as a Service)의 적정요금할인 수준 분석\n- 통행시간 및 비용변화를 중심으로 -\nDetermining Fare Discount Level for MaaS Implementation\n- Based on Time and Cost Changes -\n이 자 영*․임 이 정**․송 재 인***․황 기 연****\n* 주저자 : LG CNS 엔트루컨설팅 스마트엔지니어링 그룹\n** 공저자 : 홍익대학교 도시계획과 박사수료\n*** 공저자 : 스마트인디비쥬얼스 대표\n**** 교신저자 : 홍익대학교 도시공학과 교수\nJa Young Lee*․I Jeong Im**․Jae in song***․Kee Yeon Hwang****\n* LG CNS Co., Ltd. (Entrue Consulting)\n** Department of Urban Planning, Hongik University\n*** SmartIndividuals Pte.\n**** Department of Urban Planning, Hongik University\n†Corresponding author : Kee Yeon Hwang, keith@hongik.ac.kr\nVol.18 No.1(2019) 요 약\nFeburary, 2019\n대도시화 및 자동차의 급격한 증가에 따라 발생하는 교통문제를 해결하기 위한 대안으로\npp.1~13\nMobility as a Service(이하 Maas)의 개념이 도입되고 있으며, 유럽을 중심으로 관련한 연구와\n파일럿 프로그램 등이 추진 및 운영되고 있다. 그러나 국내의 경우 MaaS의 개념 및 도입에\n대한 정성적 연구가 주로 수행되고 있으나 도입을 위한 실증적 분석연구는 미흡한 실정이다.\nMaaS는 다양한 교통정보를 활용해 개인별로 최적화된 이동계획을 제공하여 교통수단의 대기\n시간을 줄이고 이용수요를 증가시키며 요금을 일정 수준 할인해도 수입을 줄지 않는다는 특징\n이 있다. 본 연구의 목적은 수도권에서 MaaS의 도입으로 이동수단의 통행시간 및 비용이 변하\n면 이용수요 및 수입금에 미치는 영향을 분석하고, 더 나아가 도입 타당성을 확보하기 위한\n적정 요금 할인율 수준을 도출하는데 있다. 분석을 위해 KTDB의 전국 여객 기종점 통행실태\n조사 전수화 자료를 활용하였고, 점진적 로짓모형을 활용해 MaaS 시행 전후 수담분담률 및\n수입금을 산출하였으며 최적화를 통해 적정 할인율을 산정하였다. 분석결과, 통행비용 및 통행\n시간 감소 시 각각 비승용차 수단으로 전환수요가 발생하는 것으로 분석되었으며, 통행비용에', 'metadata': {'source_id': '57845459-7165-710b-4e47-014e0832b329', 'source_file_hash': '48b23d003a93fb3381fd982b6afd61f588519932a8b861dfa89a5d9750310de6', 'source_file_name': '/home/knpu/다운로드/교통 관련 10년치 학회지_언어모델 생성용/ITS/ITS/2019/VOL 18. N 1/00018_001_1.pdf', 'source_subject': 'MaaS 도입의 적정 요금 할인율 분석', 'pages': [1], 'keywords': ['MaaS Implementation', '통행 비용', '한국ITS학회 논문지', '수준 분석', '2019 도시', 'Mobility as', '컨설팅 스마트', '얼스 대표', 'Kee Yeon', '도입 유럽']}}, vector=None, shard_key=None, order_value=None),
        ScoredPoint(id='4a1e867f-8437-f9bb-984b-df13cdaa733e', version=21, score=0.9325762208953987, payload={'page_content': '였다. 두 번째로 수단분담모형 및 점진적 로짓모델을 통해 통행비용과 통행시간 변화에 따른 수단분담률과\n수입금을 추정하고 그 결과를 기본안과 비교하였다. 마지막으로 최적 할인율을 산정하기 위한 최적화를 수행\n하였다. 이를 통해 손실이 발생하지 않는 범주 내에서 이용수요의 최대치를 갖는 적정 할인율을 산정하였다.\n<Fig. 1> Analysis process\n6 한국ITS학회논문지 제18권, 제1호(2019년 2월)\nMasS(Mobility as a Service)의 적정요금할인 수준 분석\n2. 효과분석을 위한 시뮬레이션 방법론\n앞서 살펴본 바와 같이 MaaS를 통해 수단의 통행시간 절약과 비용할인이 가능하게 된다면 승용차의 수단\n분담률은 감소하고 MaaS 이용자 수요가 증가할 것으로 기대할 수 있다. 이에 통행시간의 감소와 비용의 할\n인을 다음과 같은 방법으로 적용하였다.\n우선 통행비용은 승용차의 경우 통행시간 및 통행거리를 이용하여 존간 평균 통행속도를 산정하고 속도\n별 승용차 비용을 산출하였다. 대중교통의 경우 이동거리에 따라 요금을 정하는 수도권 통합요금제를 적용\n하여 그 비용을 추정하였다. 마지막으로 택시는 중형택시 요금체계를 적용하여 기본요금(3,000원)+추가요금\n을 거리 비례로 산정하였으며, 시간 및 지역 간 이동에 따른 추가요금은 제외하였다. 통행시간은 차외시간\n(Out-vehicle Time)과 차내시간(In-vehicle Time)으로 나누어서 수도권 교통네트워크를 이용하여 수단별로 차내\n시간, 차외시간(대기시간, 환승시간, 접근시간)을 생성하였다. KTDB처럼 택시의 대기시간 및 도보시간을 각\n5분으로 적용하였다. 또한 MaaS로 인해 감축된 통행시간은 ITF(2017)의 예측 결과(84~88%수준)를 반영하여\n평균값인 86% 수준을 적용하였다. 한편, 승용차의 경우 차외시간의 변화가 없다고 가정하였다.\n<Table 3> Travel time and cost in the Seoul Metropolitan Area\nClassification Car Taxi Bus Subway Bus+Subway\nIn-vehicle time In-Vehicle Travel Time\nWaiting\n0 5 Waiting time at the station\nTime\nTravel Time Out-\nAccess\n(min) vehicle 0 5 Walking time\nTime\nTime\nTransfer\n0 0 Time to transfer public transportation\nTime\nIn-vehicle Time - -\nReduced Time Out-of-', 'metadata': {'source_id': '4a1e867f-8437-f9bb-984b-df13cdaa733e', 'source_file_hash': '48b23d003a93fb3381fd982b6afd61f588519932a8b861dfa89a5d9750310de6', 'source_file_name': '/home/knpu/다운로드/교통 관련 10년치 학회지_언어모델 생성용/ITS/ITS/2019/VOL 18. N 1/00018_001_1.pdf', 'source_subject': 'MaaS 도입의 적정 요금 할인율 분석', 'pages': [6, 7], 'keywords': ['통행 비용', 'MaaS', '시뮬레이션 방법론', 'ITF 2017', '로짓 모델', '효과 분석', '최대 적정', 'Time Reduced', '수단 차내', 'as']}}, vector=None, shard_key=None, order_value=None),
        ScoredPoint(id='f6eceb3d-7fa3-c22f-1dfb-9c48d8f55e61', version=21, score=0.7480457400870637, payload={'page_content': 'Vol.18 No.1(2019. 2) The Journal of The Korea Institute of Intelligent Transport Systems 11\nMasS(Mobility as a Service)의 적정요금할인 수준 분석\n있다. 이는 수도권에 MaaS를 도입할 때 통행비용을 감소시키는 방향 보다는 통행시간을 최소화 할 수 있는\n이동계획을 수립하고 개개인에게 정보를 제공하는 것이 이용자 확보에 큰 영향을 끼칠 것으로 판단할 수 있\n다. 또한, MaaS 도입 시 운송업체의 수입금을 추정한 결과 도입초기에 통행비용 할인 없이 시간단축 만으로\n도 서비스 유지가 가능하다고 사료된다. 다만 분석과정에서 유발수요를 고려하지 않았기 때문에 실제 MaaS\n운영 시 본 연구의 결과보다 더 큰 수요가 창출될 수 있을 것으로 판단된다.\n본 연구의 분석결과에 따라 MaaS 이용자 수요를 확보할 수 있는 방안은 다음과 같다. 첫째, MaaS 플랫폼\n을 통한 통행시간 단축으로 초기 이용수요를 확보하는 것이다. 위에서 언급했듯 통행비용을 2.53% 할인할\n경우 운수업체의 수익금 손실이 발생하지 않으며, 이동계획 설계에 따라 통행시간을 감소시킨다면 초기 이\n용수요를 창출 할 수 있을 것으로 판단된다.\n둘째, 수집된 데이터의 활용에 따라 이용자 맞춤형 서비스를 제공하고자 한다. 초기 이용수요의 통행데이\n터를 누적하여 이용자별 선호수단, 이용패턴 등을 분석하여 향후 이동성의 편의를 제공할 수 있고 이는 다시\n이용수요를 늘리는데 기여할 것으로 사료된다.\n셋째, 이용수요의 증진효과를 통해 MaaS에 참여하는 교통수단을 확충하고자 한다. 기존 대중교통과 택시 이외에\n퍼스널 모빌리티, 카셰어링 등 다양한 수단을 확충함으로서 서비스의 질을 높여 이용률 증가를 기대할 수 있다.\n넷째, 축적된 데이터를 이용한 빅데이터 부대사업을 제안할 수 있다. 이용자의 수요가 충분히 확보가 되고\nMaaS 플랫폼이 활성화 될 경우 개인의 다양한 데이터를 수집 할 수 있을 것으로 판단된다. 빅데이터 분석을\n통해 새로운 정보를 생성하여 다양한 서비스를 개발하고 데이터 판매, 컨설팅 등 새로운 비즈니스를 통해 수\n익 창출을 기대할 수 있다. 사업적인 측면에서의 효과 이외에 서울시에서 계획 중인 단순 교통비용 할인 정\n책과 비교했을 때 승용차의 분담률 감소 및 이용수요 증진 효과 측면에서 경쟁력이 있을 것으로 기대된다.\n본 연구의 한계점은 먼저 통행시간 단축효과 기준에 대한 신뢰성 문제를 들 수 있다. 본 연구에서는 통행\n시간 단축효과를 ITF(2017) 선행연구 결과에 의존하고 있다. 또한 통행시간 감소의 기준을 모든 존에 동일하', 'metadata': {'source_id': 'f6eceb3d-7fa3-c22f-1dfb-9c48d8f55e61', 'source_file_hash': '48b23d003a93fb3381fd982b6afd61f588519932a8b861dfa89a5d9750310de6', 'source_file_name': '/home/knpu/다운로드/교통 관련 10년치 학회지_언어모델 생성용/ITS/ITS/2019/VOL 18. N 1/00018_001_1.pdf', 'source_subject': 'MaaS 도입의 적정 요금 할인율 분석', 'pages': [11, 12], 'keywords': ['MaaS 도입', '통행 비용', '빅데이터 분석', '확보 영향', '서울시 계획', '제공 초기', '이용자 수요', 'Mobility as', '2019. 2', '시간 단축']}}, vector=None, shard_key=None, order_value=None),
        ScoredPoint(id='c012e038-cae8-9656-fb57-2d86daaae300', version=21, score=0.6593218391224057, payload={'page_content': 'Vol.18 No.1(2019. 2) The Journal of The Korea Institute of Intelligent Transport Systems 7\nMasS(Mobility as a Service)의 적정요금할인 수준 분석\n<Table 3>의 수도권 통행시간 및 비용 기준을 통해 MaaS 시행 전의 통행량, 수단분담률, 통행비용, 통행시\n간을 <Table 4>와 같이 산출하였다. 실 데이터와의 유사성 검증을 위해 교통안전공단에서 제공하는 국가 대\n중교통 DB 통계자료와 비교하였으며, 서울시 평일 목적 통행 당 평균 대중교통 이용요금이 1,371원으로 매\n우 근사하게 도출되었다. 또한 목적 통행 당 대중교통 통행시간도 34.48분으로 실측값과 유사하게 나타났다.\n따라서 위의 자료를 기반으로 수단분담모형을 통해 변화하는 수단분담률 및 수익금을 산출하고자 한다.\n3. 분석모형\n본 연구에서는 수단분담모형을 통해 산출된 계수와 점진적 로짓모형을 활용하여 MaaS 시행에 따른 통행\n시간 및 비용 변화를 고려한 새로운 수단분담률을 산출하고자 한다.\n1) 수단분담모형\n수단분담모형은 교통수요 분석 또는 예측분야에서 필수적으로 사용되는 모형으로 도시, 경제, 교통 등 다\n양한 분야에서 널리 사용되고 있는 모형이다. 본 연구에서는 KTDB에서 구축한 수단분담모형을 활용하였다.\n통행목적에 따라 수단선택모형을 구축하였으며 통행빈도가 가장 높은 가정기반 통근통행 모형을 이용하였\n다. 해당 모형은 교통 수요 분석에서 일반적으로 이용되는 효용이론에 근거한 확률선택 모형 기반 로짓 모형\n(Logit Model)을 적용하고 있다. 지하철, 버스, 택시, 승용차로 총 4개의 수단으로 집계하여 모형을 구축하고,\n수단선택 모형 변수로 통행시간(차내시간+차외시간), 총 통행비용, 더미변수를 사용하였으며 그 식은 아래와\n같다.\n\ue014\ue047\ue09e\ue013 \ue048\ue09e \ue013 \ue048\ue0e8\ue0f9\ue0f1\ue0f1\ue0fd (1)\n\ue0ed \ue034 \ue0f8\ue0ed\ue0f1\ue0e9\ue0ed \ue035\ue052\ue0f4\ue0f6 cos\ue0f8\ue0ed \ue0ed', 'metadata': {'source_id': 'c012e038-cae8-9656-fb57-2d86daaae300', 'source_file_hash': '48b23d003a93fb3381fd982b6afd61f588519932a8b861dfa89a5d9750310de6', 'source_file_name': '/home/knpu/다운로드/교통 관련 10년치 학회지_언어모델 생성용/ITS/ITS/2019/VOL 18. N 1/00018_001_1.pdf', 'source_subject': 'MaaS 도입의 적정 요금 할인율 분석', 'pages': [7, 8], 'keywords': ['통행 모형', '분담 산출', 'MaaS', 'KTDB', '연구 수단', '서울시 평일', '로짓', '이용 요금', '수요 분석', 'Systems 7']}}, vector=None, shard_key=None, order_value=None),
    ]
    out = prf_expand_for_dense_and_sparse(
        query='연구에서 “적정 요금 할인율”을 산정한 방식은 무엇이며, 그 수치는 얼마였고 왜 그 수준이 최적이라고 판단되었을까요?',
        candidates=candidates,  # 1차 hybrid 상위 결과
        keywords_field="metadata.keywords",
        max_terms_sparse=8,
        max_terms_dense=4,  # dense는 더 소극적으로
    )

    dense_q = out["dense_query"]  # 임베딩용 텍스트
    sparse_q = out["sparse_query"]  # BM25 등 검색용 텍스트(구는 "..." 포함)

    import pprint
    pprint.pprint(out)