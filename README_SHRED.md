# 🌪️ SHRED V2: Granite RAG Stress Tester

`shred_v2.py` is a high-concurrency benchmarking and demonstration tool for the Granite RAG stack. It is designed to "shred" through hundreds of RAG queries simultaneously to prove the performance, stability, and concurrency of local LLM inference using **vLLM** and **Qdrant**.

## 🚀 Key Features

- **TUI Dashboard:** A real-time terminal interface built with `Rich`.
- **Live Answer Stream:** Watch the model's responses as they are generated across 128+ parallel sessions.
- **Hardware Telemetry:** Live GPU utilization, VRAM usage, and Temperature monitoring.
- **Advanced Metrics:**
    - **TTFT (Time to First Token):** Measures the "snap" and responsiveness of the model.
    - **DB Latency (Median vs. Peak):** Demonstrates the impact of "Context Switching" in vector space.
    - **vLLM Multiplier:** Quantifies the exact speedup gained by vLLM's continuous batching vs. sequential processing (Ollama-style).

## 🛠️ Requirements

- **Python 3.12+**
- **Rich** (`pip install rich`)
- **Httpx** (`pip install httpx`)
- **Qdrant Client** (`pip install qdrant-client`)
- **NVIDIA GPU** (with `nvidia-smi` available)

## 📖 How to Run

1. Ensure your Docker stack is running: `docker compose up -d`
2. Activate your virtual environment: `source .venv/bin/activate`
3. Run the shredder:
   ```bash
   python shred_v2.py
   ```

---

## 🎭 The "Live Demo" Script

Follow these steps for a high-impact presentation of the stack:

### Step 1: The "Sequential" baseline (The "Before" story)
Explain that standard local AI tools often process one request at a time. If you have 100 users, the 100th user has to wait for 99 others to finish.

### Step 2: Launch Shredder V2
Run the script and select **128 Sessions** and **500 Requests**.

### Step 3: Choose "Disparate Queries" (Mode 2)
This is the "Science" part of the demo. Tell the audience:
> *"We aren't just asking one question. We are mixing technical documentation with George Orwell's novel '1984'. Watch the **DB Peak (Switch)** metric. When the database has to jump from 'System Architecture' to 'Room 101', you'll see a latency spike as it context-switches across the vector space."*

### Step 4: Monitor the "Melt"
Point to the **Peak TPS** (Tokens Per Second). On a mobile 3080, you should see it spike over 1,000 TPS. Show the **Temperature** in the footer—notice how the GPU stays cool (often <65°C) even under massive load due to vLLM's efficiency.

### Step 5: The "Mic Drop"
Once the run finishes, the TUI will freeze. Point to the **vLLM Concurrency Multiplier**:
> *"Look at the Multiplier. If we ran these 500 queries one-by-one, we'd be standing here for 25 minutes. Because of this stack, we finished in 30 seconds. That is a **50x+ Speedup** on a single laptop."*

---

## 📊 Metric Definitions

- **Last TTFT:** The time (ms) it took for the very last request to start streaming its first token.
- **DB Latency (Med):** The median search time in Qdrant. Ignores one-time spikes to show steady performance.
- **DB Peak (Switch):** The highest latency recorded. This usually happens during a "Cold Start" or a radical shift in query topic.
- **Efficiency Multiplier:** `(Sum of all individual request times) / (Actual wall-clock time)`. This represents the "Parallelism Benefit."
