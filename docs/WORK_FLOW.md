# work flow

## 구성:
```scss
메시지 프로토콜(권장 스키마)

FastAPI 서버(WebSocket 엔드포인트)

LangGraph 기반 RAGWorkflow

Ollama/vLLM 토큰 스트리밍 연동

클라이언트(브라우저) 예시

SSE 대안 요약
```

## 1) 메시지 프로토콜

```json
// 클라이언트 → 서버
{
  "type": "query.start",
  "stream_id": "uuid-1",
  "query": "하이브리드 검색에서 alpha 설명",
  "options": {
    "model": "llama3.1:8b",
    "temperature": 0.2,
    "top_p": 0.9
  }
}

// 추가 제어(선택): pause/resume/switch_model 등
{
  "type": "control",
  "action": "pause",           // or "resume", "switch_model"
  "stream_id": "uuid-1",
  "model": "qwen2.5:14b",
  "gen": {"temperature": 0.2}
}

// 서버 → 클라이언트 (프레임 단위)
{ "type":"event", "node":"precheck", "stream_id":"uuid-1", "event":"started" }
{ "type":"json",  "node":"retrieve", "stream_id":"uuid-1", "payload": {"hits":[/* 검색결과 */]} }
{ "type":"token", "node":"generate", "stream_id":"uuid-1", "text":"하이브리드 " }
{ "type":"token", "node":"generate", "stream_id":"uuid-1", "text":"검색은..." }
{ "type":"json",  "node":"finalize", "stream_id":"uuid-1", "payload": {"answer":"...", "references":[/* 링크들 */]} }
{ "type":"done",  "node":"finalize", "stream_id":"uuid-1", "usage": {"prompt":1234, "completion":567} }
```

- type: event | token | json | done
- node: LangGraph 노드명
- stream_id: 질의/세션 단위 식별자
- text: 토큰(조각) 문자열
- payload: 구조화 JSON (검색 히트, 평가 점수, 최종 레퍼런스 등)

