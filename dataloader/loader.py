# SPDX-License-Identifier: Apache-2.0

"""
Description: Extract text and tables from PDF files, chunk the text with page tracking,
             generate embeddings using BGEM3FlagModel, and upload to Qdrant vector database.
"""

import os
import uuid

from fastapi import UploadFile
from kiwipiepy import Kiwi
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
#####################

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import SparseVectorParams
from FlagEmbedding import BGEM3FlagModel
import pdfplumber
from qdrant_client.models import VectorParams, SparseVector, Distance, models
import hashlib


from search.keyword_extractor import extract_keywords_embedrank
from models import llm_model
from pipelines.hvs_config import EmbeddingConfig, VectorStoreConfig


def extract_pdf_text_only(pdf_file_path: str | UploadFile):
    """
    pdfplumber로 텍스트만 추출하는 함수.
    페이지별 결과를 리스트(dict)로 반환하는 함수.
    """
    file_name = getattr(pdf_file_path, "filename", pdf_file_path)
    file_path = getattr(pdf_file_path, "file", pdf_file_path)
    print(f"Extracting text from {file_name}...")
    results = []

    # 전체 페이지 수 확인
    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)

    # 페이지별 순회
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_number = i + 1

            # pdfplumber 텍스트 추출
            text = page.extract_text()

            # 페이지별 결과 dict
            page_data = {
                "page_number": page_number,
                "text": text if text else "",
                "tables": None
            }

            results.append(page_data)

    return results



##################
# apt install poppler-utils
# pip install transformers
# pip install torch
# pip install pdf2image
# pip install pillow

from transformers import DonutProcessor, VisionEncoderDecoderModel, AutoTokenizer
import torch


def donut_table_parsing_pipeline(pdf_file_path):
    """
    Donut (OCR-free) 기반 table parsing pipeline.
    PDF 페이지를 이미지로 변환 후, Donut으로 구조화된 JSON을 추출.
    """

    # ✅ Donut 모델과 processor 로드
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ✅ PDF → 이미지 변환
    pages = convert_from_path(pdf_file_path)

    results = []

    for page_numer, image in enumerate(pages, start=1):
        # PIL 이미지 → processor input
        inputs = processor(image, return_tensors="pt").to(device)

        # ✅ Donut 추론
        outputs = model.generate(**inputs)

        # ✅ 결과 디코딩
        parsed_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # ✅ 페이지별 결과 저장
        page_result = {
            "page_number": page_numer,
            "parsed_text": parsed_text
        }
        results.append(page_result)

    return results



# 청크 추출
# 사전 라이브러리 설치
# pip install pdfplumber langchain sentence-transformers qdrant-client tiktoken

def get_tokenizer(model_name):
    ### 2. Tokenizer (same as embedding model)
    return AutoTokenizer.from_pretrained(model_name)


def chunk_with_multi_page_tracking(page_data, tokenizer, chunk_size=300, overlap=30):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    print("Chunking with multi-page tracking...")

    # 1. 전체 텍스트 구성 및 페이지 시작 위치 기록
    full_text = ""
    page_positions = []  # (start_pos, end_pos, page_number)
    for page in page_data:
        start = len(full_text)
        full_text += page["text"] + "\n"
        end = len(full_text)
        page_positions.append((start, end, page["page_number"]))

    # 2. chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=lambda x: len(tokenizer.encode(x, add_special_tokens=False)),
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(full_text)

    # 3. 각 chunk의 시작/끝 위치를 추정하여 포함된 페이지 리스트 생성
    result = []
    search_start_idx = 0
    for chunk in chunks:
        chunk_start = full_text.find(chunk, search_start_idx)
        if chunk_start == -1:
            # fallback: skip chunk if not found (rare but safe)
            continue
        chunk_end = chunk_start + len(chunk)
        search_start_idx = chunk_end

        # 이 chunk에 포함된 페이지 추적
        included_pages = set()
        for start, end, page_number in page_positions:
            if end <= chunk_start:
                continue  # 페이지가 chunk 이전에 끝남
            if start >= chunk_end:
                break  # 페이지가 chunk 이후에 시작함
            included_pages.add(page_number)

        result.append({
            "text": chunk,
            "pages": sorted(list(included_pages))
        })

    return result


def embedding_from_chunks(chunks, embedding_model: BGEM3FlagModel):
    """
    페이지 정보를 포함한 청크 텍스트를 임베딩하고, 각 임베딩과 메타정보를 함께 반환.
    """
    print("Generating embeddings for chunks...")

    # 텍스트만 추출
    texts = [chunk["text"] for chunk in chunks]

    # 임베딩 실행
    embeddings = embedding_model.encode(texts, return_dense=True, return_sparse=True)
    dense_vecs = embeddings["dense_vecs"]

    # 결과 구성: 임베딩 + 메타데이터
    results = []
    for i, (chunk, embedding) in enumerate(zip(chunks, dense_vecs)):
        results.append({
            "embedding": {'dense_vecs': embedding, 'lexical_weights': embeddings["lexical_weights"][i]},
            "text": chunk["text"],
            "pages": chunk["pages"]
        })

    return results


def generate_id(source: str, text: str, pages: list):
    key = f"{source}|{pages}|{text}".encode("utf-8")
    return str(uuid.UUID(hashlib.sha256(key).hexdigest()[:32]))


def gen_points_from_embeddings(embedding_results, source_data, kf):
    """
    임베딩 결과를 Qdrant 포인트 형식으로 변환.
    각 포인트에 source, text, pages 정보를 포함.
    """
    print("Generating points for Qdrant...")

    points = []
    for i, item in enumerate(embedding_results):
        #print(item)
        dense = item["embedding"]["dense_vecs"]
        indices = [int(idx) for idx in item["embedding"]["lexical_weights"].keys()]
        values = [float(val) for val in item["embedding"]["lexical_weights"].values()]
        keywords = []
        try:
            keywords = keyword_extract_func(item["text"])
        except Exception as e:
            print("keywords extract failed: %s", e)

        #print(item["embedding"])
        #print(dense)
        #print(indices)
        #print(values)

        id = generate_id(source_data["file_hash"], item["pages"], item["text"])
        point = {
            "id": id,  # 고유 ID
            "vector": {
                "dense": dense,
                "sparse": SparseVector(
                    indices=indices,
                    values=values
                )
            },
            "payload": {
                "page_content": item["text"],
                "metadata": {
                    "source_id": id,  # 고유 ID
                    "source_file_hash": str(source_data["file_hash"]),
                    "source_file_name": source_data["name"],
                    "source_subject": source_data["subject"],
                    "pages": item["pages"],
                    "keywords": keywords,
                }
            }
        }
        points.append(point)

    return points

async def upload_into_vector_db_async(client, points, collection_name):
    """
    Qdrant에 포인트를 업로드하는 함수.
    """
    print("Uploading points to Qdrant...")
    #print(points[0])

    # 컬렉션이 존재하지 않으면 생성
    if not await client.collection_exists(collection_name):
        await client.create_collection(
            collection_name=collection_name,
            on_disk_payload=False,
            vectors_config={"dense": VectorParams(
                size=len(points[0]["vector"]["dense"]),
                distance=Distance.COSINE
            )},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )

    # 포인트 업로드
    await client.upsert(
        collection_name=collection_name,
        points=points
    )

    print(f"{len(points)} chunks inserted into Qdrant.")


def hash_file_binary(file_path: str | UploadFile) -> str:
    hasher = hashlib.sha256()
    if isinstance(file_path, str):
        with open(file_path, 'rb') as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    else:
        f = file_path.file
        f.seek(0)
        while chunk := f.read(65536):
            hasher.update(chunk)
        f.seek(0)
    return hasher.hexdigest()


async def extract_source_data_async(pdf_file_path: str | UploadFile, embedding_results, llm_func):
    if isinstance(pdf_file_path, str) and not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_file_path}")
    #file_name = os.path.basename(pdf_file_path)
    file_name = getattr(pdf_file_path, "filename", pdf_file_path)
    file_hash = hash_file_binary(pdf_file_path)
    # 첫 번째 청크의 텍스트+파일명을 사용하여 주제 추출
    text = f"""
당신은 문서 제목 추출 전문가입니다.  
다음 지침을 따라 PDF 문서에서 제목을 찾아내거나 생성하세요.

1. **제목 추출 규칙**
   - 문서 전체를 분석하여, 해당 문서를 대표하는 제목을 찾습니다.
   - 표지, 목차, 서문, 저자 소개, 발행 정보 등 본문 내용이 아닌 부분은 제외합니다.
   - 제목의 특징:
     - 보통 한 줄이며 간결함.
     - 문서의 핵심 주제나 연구/보고 목적을 압축적으로 표현.
     - 불필요한 설명문이나 본문 문장이 아님.
   - 여러 후보가 있으면 가장 문서 전체를 대표하고 이해하기 쉬운 것을 선택.

2. **제목 생성 규칙**
   - 명확한 제목 후보가 없거나 문서 내 제목이 여러 개여서 판단이 어려운 경우:
     - 전체 내용을 읽고 핵심 주제 또는 결론을 한 줄로 요약하여 제목을 생성합니다.
     - 간결하고 명확하게 작성 (15자 이내가 이상적이지만, 최대 30자).
     - 독자가 문서 내용을 빠르게 파악할 수 있도록 작성.

3. **출력 형식**
   - 제목만 출력.
   - 불필요한 부연 설명, 따옴표, 번호, 마크다운 기호 사용 금지.

---

파일명: {file_name}

문서 내용:
{embedding_results[0]["text"]}

출력:"""
    subject: AIMessage = await llm_func(text)
    print(f"Extracted subject: {subject} {type(subject)}")

    return {"file_hash": file_hash, "name": file_name, "subject": subject.content.strip()}


async def test_question(collection_name, vdb, em, question):
    embedding = em.encode(question, return_dense=True, return_sparse=True)
    dense_vectors = embedding["dense_vecs"]
    sparse_vectors = embedding["lexical_weights"]
    indices = [int(idx) for idx in sparse_vectors.keys()]
    values = [float(val) for val in sparse_vectors.values()]
    print("dense:", dense_vectors)
    print("sparse:", sparse_vectors)
    print("sparse.indices:", indices)
    print("sparse.values:", values)
    if sparse_vectors:
        results = await vdb.query_points(
            collection_name,
            prefetch=models.Prefetch(query=dense_vectors, using="dense", limit=10),
            query=SparseVector(indices=indices, values=values),
            using="sparse",
            with_payload=True,
            limit=10,
        )
    else:
        results = await vdb.query_points(
            collection_name,
            query=dense_vectors,
            using="dense",
            with_payload=True,
            limit=10,
        )
    print(results)


async def get_subject_from_llm_async(llm: BaseChatModel, text):
    return await llm.ainvoke(text)


def save_pdf_into_storage(pdf_file_path: str | UploadFile, param):
    # 지정된 object storage에 PDF 파일을 저장하는 함수 (MinIO, AWS S3 등)
    file_name = getattr(pdf_file_path, "filename", pdf_file_path)
    print(f"Saving PDF file {file_name} into storage with param: {param}")
    # s3_path = f"pdf_storage/{os.path.basename(pdf_file_path)}"
    # import boto3
    # s3 = boto3.client('s3')
    # bucket_name = "your-bucket-name"  # S3 버킷 이름
    # try:
    #     s3.upload_file(pdf_file_path, bucket_name, s3_path)
    #     print(f"PDF file saved to S3: {s3_path}")
    # except Exception as e:
    #     print(f"Failed to save PDF file to S3: {e}")


async def run_async(collection_name, pdf_file_path: str| UploadFile, embedding_model, tokenizer_model_name, vectordb, llm_func, keyword_extract_func):
    data = extract_pdf_text_only(pdf_file_path)
    if not data:
        print("‼️ No text extracted from the PDF: ", getattr(pdf_file_path, "filename", pdf_file_path))
        return
    chunks = chunk_with_multi_page_tracking(data, get_tokenizer(model_name=tokenizer_model_name), chunk_size=500, overlap=100)
    embedding_results = embedding_from_chunks(chunks, embedding_model=embedding_model)
    source_data = await extract_source_data_async(pdf_file_path, embedding_results, llm_func)
    points = gen_points_from_embeddings(embedding_results, source_data, keyword_extract_func)
    await upload_into_vector_db_async(vectordb, points, collection_name)
    save_pdf_into_storage(pdf_file_path, source_data["file_hash"])



if __name__ == "__main__":
    pdf_path = "/home/knpu/다운로드/교통 관련 10년치 학회지_언어모델 생성용/ITS/ITS/2019/"
    pdf_path = "/home/knpu/다운로드/교통 관련 10년치 학회지_언어모델 생성용/ITS/ITS/2019/VOL 18. N 1/00018_001_0.pdf"
    cfg_path = "../config/rag.yaml"
    em_cfg = EmbeddingConfig(cfg_path)
    vs_cfg = VectorStoreConfig(cfg_path)
    tokenizer_model_name = em_cfg.tokenizer
    em = BGEM3FlagModel(em_cfg.bgem3, use_fp16=True)
    kiwi = Kiwi(num_workers=4)  # 형태소 분석기 초기화
    # Qdrant 클라이언트 초기화
    vdb = AsyncQdrantClient(url=vs_cfg.qdrant_url)  # Qdrant 서버 URL
    collection_name = vs_cfg.collection

    llm_cfg = llm_model._load_yaml("../config/llm.yaml")
    tiers = llm_model._get_llm_tiers(llm_cfg)
    llm = llm_model._build_single_llm(tiers["small"])

    # make a variable which is get_subject_from_llm(mlm), so that I can call it later with any text
    get_subject_from_llm_func = lambda text: get_subject_from_llm_async(llm, text)
    keyword_extract_func = lambda text: extract_keywords_embedrank(em=em, kiwi=kiwi, text=text, top_n=10, diversity=0.6, ngram_range=(1,2))


    # extract file_path from directory_path
    async def _run_main():
        for root, dirs, files in os.walk(pdf_path):
            print(root, dirs, files)
            for file in files:
                if file.endswith(".pdf"):
                    pdf_file_path = os.path.join(root, file)
                    await run_async(collection_name, pdf_file_path, em, tokenizer_model_name, vdb, get_subject_from_llm_func, keyword_extract_func)
    import asyncio
    asyncio.run(_run_main())
    #print("✅ All PDFs processed successfully.")
    asyncio.run(test_question(collection_name, vdb, em, "왜 통행시간 단축이 더 효과적이야?"))
