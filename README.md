# Granite RAG Stack

A self-hosted, performance-optimized Retrieval-Augmented Generation (RAG) stack built around IBM's Granite models. This stack is designed to run on consumer-grade NVIDIA hardware, providing a complete pipeline from PDF ingestion to a full-featured Web UI.

## Features

- **Local Inference:** Powered by `vLLM` for high-throughput LLM and embedding generation.
- **Granite Models:** Uses `ibm-granite/granite-4.0-h-micro` for chat and `granite-embedding-125m-english` for vectorization.
- **Vector Database:** Qdrant for efficient similarity search and metadata management.
- **Two Ways to Chat:**
  - **Open WebUI:** A polished, ChatGPT-like browser interface.
  - **CLI Chat:** A lightweight Python script for terminal-based interaction with citations.
- **Automated Ingestion:** Python script to process PDFs with automatic chunking and page-level metadata.

## Architecture

- **vLLM (Port 8000):** Serves the generative model.
- **vLLM Embed (Port 8001):** Serves the embedding model.
- **Qdrant (Port 6333):** Stores document vectors.
- **Open WebUI (Port 3000):** The primary user interface.

## Prerequisites

- **OS:** Linux (tested on Ubuntu)
- **Hardware:** NVIDIA GPU with 8GB+ VRAM recommended.
- **Software:** 
  - [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  - Python 3.12+

## Getting Started

### 1. Clone and Setup Environment
```bash
git clone <your-repo-url>
cd vllm-stack

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Stack
```bash
docker compose up -d
```
*Note: The first run will download several GBs of Docker images and model weights (cached in `./hf-cache`).*

### 3. Ingest Your Documents
Drop your `.pdf` files into the project root and run:
```bash
python ingest.py
```

### 4. Start Chatting
- **Web UI:** Open `http://localhost:3000` in your browser.
- **CLI:** Run `python ragchat.py` for a terminal interface.

## Configuration

- **VRAM Usage:** You can tune the `--gpu-memory-utilization` flags in `docker-compose.yml` to fit your specific GPU.
- **Chunking:** Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` in `ingest.py` to change how documents are processed.

## License
[Specify License, e.g., Apache 2.0]
