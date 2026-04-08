# Granite RAG Stack

A self-hosted Retrieval-Augmented Generation (RAG) stack optimized for consumer hardware using IBM Granite models. This project provides both a CLI-based RAG chat and a full Web UI.

## Project Overview

The stack orchestrates multiple containers to provide a complete AI pipeline:
- **Inference:** vLLM serves the Granite 4.0 Micro LLM and Granite Embedding models.
- **Vector Storage:** Qdrant handles document storage and similarity search.
- **Frontend:** Open WebUI provides a ChatGPT-like interface configured to use the local vLLM instances.
- **Tooling:** Python scripts for PDF ingestion and local CLI interaction.

## Main Technologies

- **LLM:** `ibm-granite/granite-4.0-h-micro` (via vLLM)
- **Embedding:** `ibm-granite/granite-embedding-125m-english` (via vLLM and local CPU)
- **Vector Database:** Qdrant
- **UI:** Open WebUI
- **PDF Processing:** PyMuPDF (fitz)
- **Environment:** Docker Compose, Python 3.12 (venv)

## Architecture & Ports

- **vLLM Chat (8000):** OpenAI-compatible API for the Granite LLM.
- **vLLM Embed (8001):** Dedicated embedding endpoint for high-throughput ingestion.
- **Qdrant (6333):** Vector database API and Dashboard.
- **Open WebUI (3000):** Main user interface.

## Usage

### 1. Start the Infrastructure
Ensure you have the NVIDIA Container Toolkit installed.
```bash
docker compose up -d
```

### 2. Ingest Documents
Place `.pdf` files in the root directory and run the ingestion script. This will chunk the documents and store them in the `granite_pdf_test` collection in Qdrant.
```bash
# Ensure venv is active
source .venv/bin/activate
python ingest.py
```

### 3. Chat via CLI
For a quick terminal-based interaction with your documents:
```bash
python ragchat.py
```

### 4. Web UI
Visit `http://localhost:3000`. It is pre-configured to use the vLLM services for both chat and RAG embeddings. (Auth is disabled by default).

## Development Conventions

- **Model Caching:** All models are cached in `./hf-cache` to avoid repeated downloads.
- **Stability:** `VLLM_USE_V1=0` is set in the compose file for better compatibility with older/consumer GPUs.
- **Chunking:** `ingest.py` uses a chunk size of 400 characters with 50 character overlap to stay within the 512-token limit of the embedding model.
- **Metadata:** Ingestion captures both the source filename and the page number for accurate citations during chat.
- **Working Files:** `.working` and `.old` files are kept as reference for known-good configurations.

## Maintenance

- **Wiping Data:** To reset the vector database, you can either delete the collection via the Qdrant API or remove the `qdrant_data` volume.
- **VRAM Tuning:** Adjust `--gpu-memory-utilization` in `docker-compose.yml` if you encounter Out-Of-Memory (OOM) errors. The micro model is currently set to `0.8` (80%) and the embedding model to `0.1` (10%).
