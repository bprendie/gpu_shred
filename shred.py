import asyncio
import httpx
import time
import random
import statistics
from qdrant_client import AsyncQdrantClient

# --- Configuration ---
CHAT_URL = "http://localhost:8000/v1/chat/completions"
EMBED_URL = "http://localhost:8001/v1/embeddings"
MODEL_CHAT = "ibm-granite/granite-4.0-h-micro"
EMBED_MODEL = "ibm-granite/granite-embedding-125m-english"
QDRANT_URL = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "granite_pdf_test"

# Sample Technical Queries to Stress Retrieval and LLM
QUERIES = [
    "What are the main architectural constraints mentioned?",
    "How does the system handle high-throughput ingestion?",
    "Explain the security model described in the documents.",
    "What are the hardware requirements for deployment?",
    "Compare the performance benchmarks for different models.",
    "Describe the error handling mechanism for network failures.",
    "How is metadata handled during the chunking process?",
    "What are the limitations of the current implementation?",
    "List all technical dependencies and their versions.",
    "Explain the data retention policy for the vector database."
]

# Shared metrics
total_tokens = 0
total_requests = 0
target_requests = 0
max_tps = 0
request_times = []
start_time = 0

async def get_embedding_async(client, text):
    """Fetch query embedding from vLLM on port 8001."""
    payload = {"model": EMBED_MODEL, "input": [text]}
    res = await client.post(EMBED_URL, json=payload, timeout=30)
    res.raise_for_status()
    return res.json()["data"][0]["embedding"]

async def retrieve_context_async(q_client, query_vector, limit=3):
    """Search Qdrant for context."""
    search_result = await q_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True
    )
    context_blocks = [hit.payload.get("text", "") for hit in search_result.points]
    return "\n---\n".join(context_blocks)

async def chat_with_vllm_async(client, prompt, context):
    """Send augmented prompt to vLLM on port 8000."""
    global total_tokens
    system_prompt = f"Use the context to answer the question briefly.\n\nCONTEXT:\n{context}"
    payload = {
        "model": MODEL_CHAT,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 128 # Keep responses concise but stressful
    }
    
    start_req = time.perf_counter()
    res = await client.post(CHAT_URL, json=payload, timeout=60)
    res.raise_for_status()
    end_req = time.perf_counter()
    
    data = res.json()
    tokens = data["usage"]["completion_tokens"]
    total_tokens += tokens
    request_times.append(end_req - start_req)
    return tokens

async def session_worker(worker_id, q_client, http_client, num_sessions):
    """A single concurrent session loop."""
    global total_requests
    while total_requests < target_requests:
        try:
            query = random.choice(QUERIES)
            
            # 1. Embed
            vector = await get_embedding_async(http_client, query)
            
            # 2. Retrieve
            context = await retrieve_context_async(q_client, vector)
            
            # 3. Chat
            tokens = await chat_with_vllm_async(http_client, query, context)
            
            total_requests += 1
            
        except Exception as e:
            # Silence errors to avoid cluttering benchmark, but increment requests
            pass
        
        # Small sleep to prevent tight-looping if vLLM is very fast
        await asyncio.sleep(0.05)

async def monitor(num_sessions):
    """Prints live metrics every second on a single updating line."""
    global max_tps
    print(f"\n{'Time':<6} | {'Progress':<12} | {'Req/s':<8} | {'Tokens/s':<10} | {'Latency':<8} | {'Total Req'}")
    print("-" * 75)
    
    last_tokens = 0
    last_requests = 0
    
    while total_requests < target_requests:
        await asyncio.sleep(1)
        elapsed = time.time() - start_time
        
        current_tokens = total_tokens
        current_requests = total_requests
        
        tps = current_tokens - last_tokens
        rps = current_requests - last_requests
        max_tps = max(max_tps, tps)
        
        percent = (current_requests / target_requests) * 100
        progress_bar = f"[{'#' * int(percent // 10)}{'.' * (10 - int(percent // 10))}]"
        
        avg_latency = statistics.mean(request_times[-20:]) if request_times else 0
        
        # Use \r to return to the start of the line and end="" to avoid a newline
        output = f"\r{int(elapsed):<6}s | {progress_bar} {int(percent):>3}% | {rps:<8} | {tps:<10} | {avg_latency:.2f}s  | {current_requests}/{target_requests}"
        print(output, end="", flush=True)
        
        last_tokens = current_tokens
        last_requests = current_requests
    print() # Final newline when finished

async def main():
    global start_time, target_requests
    print("=== SHRED.PY - STRESS TESTING GRANITE RAG STACK ===")
    
    try:
        choice = input("Concurrent Sessions (1, 16, 64, 128, 256) [default 16]: ")
        num_sessions = int(choice) if choice else 16
        
        limit_choice = input("Total Requests to Shred [default 500]: ")
        target_requests = int(limit_choice) if limit_choice else 500
    except ValueError:
        num_sessions = 16
        target_requests = 500

    print(f"\n[WARMUP] Starting {num_sessions} sessions to shred {target_requests} requests...")
    
    q_client = AsyncQdrantClient(host=QDRANT_URL, port=QDRANT_PORT)
    limits = httpx.Limits(max_keepalive_connections=num_sessions, max_connections=num_sessions + 10)
    
    async with httpx.AsyncClient(limits=limits) as http_client:
        start_time = time.time()
        
        # Start monitor
        monitor_task = asyncio.create_task(monitor(num_sessions))
        
        # Start workers
        workers = [
            asyncio.create_task(session_worker(i, q_client, http_client, num_sessions))
            for i in range(num_sessions)
        ]
        
        await asyncio.gather(*workers)
        monitor_task.cancel()
        
        total_time = time.time() - start_time
        print(f"\n[COMPLETE] Shredded {target_requests} requests in {total_time:.2f}s")
        print(f"Final Average: {total_tokens / total_time:.2f} tokens/sec")
        print(f"Peak Performance: {max_tps} tokens/sec")
        
        await q_client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[STOPPED] Benchmarking complete.")
