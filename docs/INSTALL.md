# RAG 솔루션 설치 가이드

## 시스템 요구사항

### 소형 트래픽용 하드웨어 구성 (동시 사용자 1~2명 정도)
- 데몬 구성
  - RAG Engine 서버
  - Model Serving 서버
  - 벡터 DB 서버
  - WEB 서버
- 장비 스펙 (1대)
    - 운영체제: Linux
    - CPU: intel x86-64 계열, 최소 4코어
    - 메모리: 최소 16GB RAM
    - GPU 종류: NVIDIA 계열
    - GPU 메모리: 최소 32GB VRAM
    - 스토리지: 최소 100GB SSD

**참고**: 더 많은 동시사용자 처리 시스템 구성은 별도 구매 문의 바랍니다.


## 사전 설치 소프트웨어 목록

- NVIDIA Driver Version: ***570.158.01 이후 버전 필수***
- CUDA Version: ***12.8 이후 버전 필수, NVIDIA Drivder 버전과 호환 필수***
- Docker Version: ***28.1.1 이후 버전 필수***
- OLlama Version: 최신 (0.11.4 이후 버전)

## 사전 소프트웨어 설치 방법

### OS 확인 방법
```bash
cat /etc/os-release
```

### CPU 확인 하는 방법
```bash
uname -m
```

### NVIDIA Driver (GPU 서버용)
- nvidia-smi 명령어로 설치 확인 가능
- 하드웨어 및 OS에 따라 설치 방법이 다르므로 [NVIDIA 공식 설치 가이드](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)를 참고하세요.
- [공식 NVIDIA 드라이버 다운로드](https://www.nvidia.com/en-us/drivers/) 
- 간략 설치 방법 예시 (Ubuntu 기준)
```bash
# NVIDIA 장비 확인
lspci | grep -i NVIDIA

# 드라이버 설치를 위한 준비
sudo apt-get update -y
sudo apt-get install -y ubuntu-drivers-common

# 드라이버 버전 확인
ubuntu-drivers devices

# 드라이버 설치
sudo apt install nvidia-driver-535

# 설치 후 재부팅
sudo reboot

# 설치 확인
nvidia-smi
```

### CUDA (GPU 서버용)
- nvcc -V 명령어로 설치 확인 가능
- ***NVIDIA 드라이버와 호환되는 버전을 설치해야 함***
- [호환성 표](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html)
- 자세한건 NVIDIA 공식 설치 가이드 참고
- [NVIDIA CUDA Toolkit 다운로드](https://developer.nvidia.com/cuda-toolkit-archive)
- 간략 설치 방법 예시
  - 위 다운로드 링크에서 적합한 버전 선택 후 화면에 출력된 설치 방법에 따라 설치
  - 설치 후 bashrc에 환경변수 추가
  - nvcc -V 명령어로 설치 확인

### Docker (모든 서버용)
- Amazon Linux 2023 / RHEL / CentOS인 경우
```bash
sudo dnf install -y docker
sudo systemctl enable docker
sudo systemctl start docker
```

- Amazon Linux 2인 경우
```bash
sudo amazon-linux-extras install docker -y
sudo systemctl enable docker
sudo systemctl start docker
```

- 일반 계정으로 사용하기 위한 명령어
```bash
sudo usermod -aG docker $USER
newgrp docker
```

- 기타 OS는 [Docker 공식 설치 가이드](https://docs.docker.com/engine/install/)를 참고하세요.


### OLlama (Model Serving 서버용)
- Docker가 사전에 설치 되어 있어야 합니다.
- ***11434 포트의 통신이 가능해야 합니다.***
- 기타 설치 정보는 [Ollama 공식 사이트](https://ollama.com/download/linux), [공식 설치 방법](https://github.com/ollama/ollama/blob/main/docs/linux.mdx)을 참고하세요.
```bash
sudo apt install curl -y
sudo curl -fsSL https://ollama.com/install.sh | sh
ollama --version

sudo useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama
sudo usermod -a -G ollama $(whoami)

vi /etc/systemd/system/ollama.service

# 아래 내용으로 편집 후 저장
# ExecStart 경로는 ollama 설치 경로에 맞게 수정 필요
# which ollama 명령어로 설치 경로 확인 가능
---
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
Environment=OLLAMA_HOST=0.0.0.0:11434
Environment=OLLAMA_KEEP_ALIVE=-1
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=$PATH"

[Install]
WantedBy=multi-user.target
---

# 서비스 등록 및 시작
sudo systemctl daemon-reload
sudo systemctl enable ollama
```

## LLM 구성요소 설치 방법
- 제공된 압축파일을 서버에 업로드 후 압축 해제
- 디스크 용량을 확인하세요. 최소 50GB 이상의 여유 공간 필요
```bash
# Ubuntu / Debian 계열
sudo apt update
sudo apt install zstd -y

# Amazon Linux 2023 / RHEL / CentOS 계열
sudo dnf install zstd -y

# 압축 해제
tar -xvf myapp-v2.tar.zst -I "zstd -T0"
```
- 압축 파일 내용
```scss
/myapp-v2                               // 압축파일 루트 경로
|____myapp-v2.tar.md5                   // RAG 솔루션 압축파일 MD5 체크섬
|____myapp-v2.tar                       // RAG 솔루션 압축파일
|____model
| |____gemma-3-27b-it-Q5_K_M.gguf.md5   // LLM모델 파일 MD5 체크섬
| |____gemma-3-27b-it-Q5_K_M.gguf       // LLM 모델 파일
|____config                             // 환경 설정 디렉토리
| |____prompts.yaml
| |____rag.yaml
| |____llm.yaml
```

### LLM Model 다운로드 (Model Serving 서버용)
```bash
cd model
ollama run gemma-3-27b-it-Q5_K_M
```

### Vector DB 설치 (Qdrant)
- Docker가 사전에 설치 되어 있어야 합니다.
- ***6333, 6334, 6335*** 포트의 통신이 가능해야 합니다.
- 기타 설치 정보는 [Qdrant 공식 사이트](https://qdrant.tech/documentation/guides/installation)를 참고하세요.
```bash
# Qdrant Docker 이미지 다운로드
docker pull qdrant/qdrant

# 데이터 저장용 디렉토리 생성 (별도의 경로가 있으면 해당 경로로 바꾸기)
sudo mkdir -p /app/qdrant/data
sudo chown $(whoami):$(whoami) /app/qdrant/data

# Qdrant 컨테이너 실행 (별도의 경로가 있으면 해당 경로로 바꾸기)
docker run -d --restart always \
  -p 6333:6333 -p 6334:6334 \
  -v /app/qdrant/data:/qdrant/storage \
  qdrant/qdrant

```

## RAG 엔진 설치
```bash
# 기존에 압축 해제한 디렉토리 (deploy)로 이동
cd deploy

# config 디렉토리가 있는지 확인하기
ls -al

# RAG 이미지 구동
docker run -d --restart unless-stopped --shm-size=4gb --gpus all \
  --name myapp_container -p 9000:9000 \
  -v ./config:/app/config \
  myapp:latest
```
구동 오류시 아래와 같이 컨테이너 삭제 후 다시 시도하세요.
```bash
# docker: Error response from daemon: Conflict. The container name "/myapp_container" is already in use by container "3d5726996378f83adaaba043bf7626ab820f3f9143cf0f4a4d618cefa3356173". You have to remove (or rename) that container to be able to reuse that name.

docker ps -a
docker rm -f [CONTAINER ID]
```

구동 확인 방법
```bash
docker ps
docker logs myapp_container -f

# 아래처럼 나오면 성공
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9000 (Press CTRL+C to quit)
```

## 내부 테스트
- RAG 엔진이 정상적으로 작동하는지 테스트 하는 방법
  - http://[RAG_ENGINE_SERVER_IP]:9000/static/rag.html 접속
  - 접속후 "진행로그" 영역에서 "[ws] connected" 메시지 확인
  - 질문 입력 후 "전송" 버튼 클릭
  - docker logs myapp_container -f 명령어로 RAG 엔진 로그에 오류가 없는지 확인
- Vector DB 정상 동작 확인 방법
  - http://[VECTOR_DB_SERVER_IP]:6333/dashboard 접속
  - Qdrant Swagger UI가 정상적으로 출력되는지 확인
  - Collection 들의 configuration이 정상인지 확인 (dense-1024-Cosine, sparse-Sparse)
  - Configuration이 잘못된 경우 "..."(Action) 을 눌러서 Delete 후 Collection 재생성 필요

## 네트워크 구성도 (방화벽 설정)
![네트워크-구성.png](%E1%84%82%E1%85%A6%E1%84%90%E1%85%B3%E1%84%8B%E1%85%AF%E1%84%8F%E1%85%B3-%E1%84%80%E1%85%AE%E1%84%89%E1%85%A5%E1%86%BC.png)