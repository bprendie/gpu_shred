# GPU Shred: Granite RAG Parallelism Demo

This project is a high-performance demonstration of how to run massively parallel RAG (Retrieval-Augmented Generation) workloads on consumer-grade NVIDIA hardware. By leveraging **vLLM** and **IBM Granite** models, this stack showcases how local inference can achieve enterprise-level throughput by breaking the "one-request-at-a-time" bottleneck.

**The Core Metric:** As you scale concurrent sessions, your **Tokens Per Second (TPS)** will skyrocket. While serial processing waits for each word to generate, vLLM's continuous batching fills the GPU's "dead air" with parallel requests, leading to 10x-30x efficiency gains.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- **OS:** Linux (Ubuntu recommended)
- **Hardware:** NVIDIA GPU (8GB+ VRAM) with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.
- **Software:** Docker & Docker Compose.

### 2. Environment Setup
We recommend using a Python virtual environment to keep your system clean:

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install the necessary libraries
pip install -r requirements.txt
```

### 3. Launch the Infrastructure
```bash
docker compose up -d
```
*Note: The first run will download several GBs of model weights into `./hf-cache`.*

---

## 🚀 The Main Event: `shred_v2.py`

`shred_v2.py` is a specialized stress-testing TUI (Terminal User Interface) designed to push the Granite models to their absolute limit.

### What it does:
- **Massive Concurrency:** Spawns up to 256 simultaneous RAG sessions.
- **Real-Time Stress:** Each session performs a full RAG loop: 
    1.  Embeds a random query.
    2.  Searches the Qdrant vector database.
    3.  Streams a response from the Granite-4.0-h-micro model.
- **Data Diversity & Context Switching:** The demo utilizes a mix of dense technical manuals and George Orwell's *1984*. By switching between technical architecture queries and dystopian narrative concepts, the script forces massive context switching in both the vLLM KV-cache and the Qdrant retrieval engine. This ensures the benchmarks reflect genuine "cold" processing power rather than relying on repetitive cache hits.
- **Live Metrics:** Monitors TPS (Tokens/sec), RPS (Requests/sec), TTFT (Time to First Token), and DB Latency.
- **GPU Telemetry:** Direct integration with `nvidia-smi` to show VRAM and Core utilization during the "shred."
- **Efficiency Multiplier:** At the end of the run, it calculates how much faster the workload completed compared to traditional serial processing (Ollama-style).

### How to run:
```bash
# Ensure the stack is running first
docker compose up -d

# Run the shredder
python shred_v2.py
```

---

## 💬 Quick Interaction (`ragchat.py`)

If you want a "quick and dirty" way to talk to your documents without the full stress-test TUI, use `ragchat.py`. It provides a simple command-line interface for direct RAG queries, making it ideal for verifying that your ingestion worked or for performing individual lookups.

---

## 📥 Data Ingestion (`ingest.py`)

Before you can shred, you need data. This script processes any PDFs in the root directory.

### How it works:
1.  **Chunking:** Breaks PDFs into 400-character segments with a 50-character overlap.
2.  **GPU Vectorization:** Sends chunks to the `vllm_embed` container (Port 8001).
3.  **Metadata Tagging:** Captures the source filename and the specific page number for every chunk.
4.  **Upsert:** Stores the 768-dimensional vectors in Qdrant.

### How to run:
1.  Drop `.pdf` files into the root folder.
2.  Execute the script:
    ```bash
    python ingest.py
    ```

---

## 🔍 Qdrant: Vector Storage & Visualization

Qdrant handles the heavy lifting of semantic search.

### Accessing the Dashboard
You can monitor your collections and manually query points through the web interface:
- **URL:** [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

### Visualizing Embeddings
One of Qdrant's most powerful features is the built-in **Visualize** tool.
1.  Open the Dashboard.
2.  Click on the `granite_pdf_test` collection.
3.  Click the **Visualize** tab.
4.  This generates a 2D/3D map of your document chunks. You can literally *see* the semantic clusters where your PDFs discuss similar topics.

---

## 🛠️ Architecture

- **vLLM (Port 8000):** Generative Model (`granite-4.0-h-micro`).
- **vLLM Embed (Port 8001):** Embedding Model (`granite-embedding-125m-english`).
- **Qdrant (Port 6333):** Vector Database.
- **Open WebUI (Port 3004):** A full-featured UI for standard RAG chat.

---

## ⚡ Performance Tuning
In `docker-compose.yml`, you can adjust the `--gpu-memory-utilization` flag.
- **Micro Model:** Currently set to `0.8` (80% VRAM).
- **Embedding Model:** Currently set to `0.1` (10% VRAM).

If you have a 16GB+ card, you can increase these or swap the Micro model for the "Tiny" or "Base" variants for higher quality at the cost of some throughput.
