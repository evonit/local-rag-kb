# SPDX-License-Identifier: Apache-2.0

"""
Description: Test script for embedding functionalities in KnowledgeOrbit.AI project.
"""

from transformers import AutoTokenizer

def tokenize_text(model_name, text):
    # 실제 Qdrant에 넣을 때 사용한 임베딩 모델의 이름을 넣어야 함
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    print("모델: ", model_name)
    print("입력 문장:", text)
    print("토큰:", tokens)
    print("토큰 ID:", ids)
    print("토큰 개수:", len(tokens))


if __name__ == "__main__":
    text = "연구에서 “적정 요금 할인율”을 산정한 방식은 무엇이며, 그 수치는 얼마였고 왜 그 수준이 최적이라고 판단되었을까요?"
    tokenize_text("nlpai-lab/kure-v1", text)
    tokenize_text("dragonkue/BGE-m3-ko", text)
    tokenize_text("BAAI/bge-m3", text)
    tokenize_text("skt/A.X-Encoder-base", text)


