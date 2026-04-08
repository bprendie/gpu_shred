import asyncio
import httpx
import time
import random
import statistics
import json
import subprocess
from datetime import datetime
from collections import deque
from qdrant_client import AsyncQdrantClient

# Rich TUI Imports
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.text import Text
from rich.align import Align

# --- Configuration ---
CHAT_URL = "http://localhost:8000/v1/chat/completions"
EMBED_URL = "http://localhost:8001/v1/embeddings"
MODEL_CHAT = "ibm-granite/granite-4.0-h-micro"
EMBED_MODEL = "ibm-granite/granite-embedding-125m-english"
QDRANT_URL = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "granite_pdf_test"

# --- Query Sets ---
TECH_QUERIES = [
    "What are the main architectural constraints mentioned?",
    "How does the system handle high-throughput ingestion?",
    "Explain the security model described in the documents.",
    "What are the hardware requirements for deployment?",
    "Compare the performance benchmarks for different models.",
    "Describe the error handling mechanism for network failures.",
    "How is metadata handled during the chunking process?",
    "What are the limitations of the current implementation?",
    "Explain the data retention policy for the vector database.",
    "List all technical dependencies and their versions."
]

DYSTOPIAN_QUERIES = [
    "Who is Big Brother and what is his role?",
    "Explain the concept of Doublethink.",
    "What is Newspeak and why is it being implemented?",
    "Describe the role of the Thought Police.",
    "What happens in Room 101?",
    "What does the slogan 'War is Peace' mean?",
    "Who is Winston Smith and what is his job at the Ministry of Truth?",
    "What is the Junior Anti-Sex League?",
    "Describe the atmosphere of Victory Mansions.",
    "What are the Three Slogans of the Party?"
]

# Shared Global State
state = {
    "total_tokens": 0,
    "total_requests": 0,
    "target_requests": 500,
    "num_sessions": 16,
    "start_time": 0,
    "last_tokens": 0,
    "last_requests": 0,
    "max_tps": 0,
    "total_request_time": 0,
    "last_request_time": 0,
    "request_times": deque(maxlen=20),
    "max_db_latency": 0,
    "total_db_time": 0,
    "total_db_calls": 0,
    "retrieval_times": deque(maxlen=30),
    "ttft_times": deque(maxlen=50),
    "last_ttft": 0,
    "responses": deque(maxlen=10),
    "gpu_util": "0",
    "gpu_temp": "0",
    "gpu_mem_used": "0",
    "gpu_mem_total": "0",
    "qdrant_points": 0,
    "qdrant_vectors": 0,
    "active_queries": [],
    "query_mode": "Similar",
    "is_running": True
}

async def get_gpu_stats():
    """Fetch GPU metrics via nvidia-smi."""
    while state["is_running"]:
        try:
            cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"
            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            if stdout:
                parts = stdout.decode().strip().split(",")
                state["gpu_util"] = parts[0].strip()
                state["gpu_mem_used"] = parts[1].strip()
                state["gpu_mem_total"] = parts[2].strip()
                state["gpu_temp"] = parts[3].strip()
        except Exception:
            pass
        await asyncio.sleep(1)

async def get_qdrant_stats(q_client):
    """Fetch collection stats from Qdrant."""
    while state["is_running"]:
        try:
            info = await q_client.get_collection(COLLECTION_NAME)
            state["qdrant_points"] = info.points_count
            state["qdrant_vectors"] = info.indexed_vectors_count
        except Exception:
            pass
        await asyncio.sleep(2)

async def get_embedding_async(client, text):
    payload = {"model": EMBED_MODEL, "input": [text]}
    res = await client.post(EMBED_URL, json=payload, timeout=30)
    res.raise_for_status()
    return res.json()["data"][0]["embedding"]

async def retrieve_context_async(q_client, query_vector, limit=3):
    start_search = time.perf_counter()
    search_result = await q_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True
    )
    end_search = time.perf_counter()
    duration = end_search - start_search
    state["retrieval_times"].append(duration)
    state["max_db_latency"] = max(state["max_db_latency"], duration)
    state["total_db_time"] += duration
    state["total_db_calls"] += 1
    
    context_blocks = [hit.payload.get("text", "") for hit in search_result.points]
    return "\n---\n".join(context_blocks)

async def chat_with_vllm_async(client, prompt, context):
    system_prompt = f"Answer briefly based on context.\n\nCONTEXT:\n{context}"
    payload = {
        "model": MODEL_CHAT,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 128,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    
    start_req = time.perf_counter()
    ttft = None
    full_content = ""
    tokens = 0
    
    async with client.stream("POST", CHAT_URL, json=payload, timeout=60) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    
                    if not ttft and chunk["choices"] and chunk["choices"][0]["delta"].get("content"):
                        ttft = time.perf_counter() - start_req
                        state["ttft_times"].append(ttft)
                        state["last_ttft"] = ttft
                    
                    if chunk["choices"]:
                        content = chunk["choices"][0]["delta"].get("content", "")
                        full_content += content
                    
                    if "usage" in chunk and chunk["usage"]:
                        tokens = chunk["usage"]["completion_tokens"]
                except:
                    continue
    
    end_req = time.perf_counter()
    if tokens == 0:
        tokens = len(full_content.split())
    
    req_duration = end_req - start_req
    state["total_tokens"] += tokens
    state["total_request_time"] += req_duration
    state["last_request_time"] = req_duration
    state["request_times"].append(req_duration)
    
    ts = datetime.now().strftime("%H:%M:%S")
    display_content = full_content.replace("\n", " ").strip()
    formatted_resp = f"[cyan]{ts}[/cyan] [bold green]➜[/bold green] {display_content[:160]}..."
    state["responses"].append(formatted_resp)
    
    return tokens

async def session_worker(worker_id, q_client, http_client):
    while state["total_requests"] < state["target_requests"]:
        try:
            query = random.choice(state["active_queries"])
            vector = await get_embedding_async(http_client, query)
            context = await retrieve_context_async(q_client, vector)
            await chat_with_vllm_async(http_client, query, context)
            state["total_requests"] += 1
        except Exception:
            pass
        await asyncio.sleep(0.01)

def make_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3)
    )
    layout["main"].split_row(
        Layout(name="stats", ratio=1),
        Layout(name="responses", ratio=2)
    )
    return layout

def get_stats_table(is_final=False):
    elapsed = time.time() - state["start_time"]
    tps = state["total_tokens"] - state["last_tokens"]
    rps = state["total_requests"] - state["last_requests"]
    state["max_tps"] = max(state["max_tps"], tps)
    
    state["last_tokens"] = state["total_tokens"]
    state["last_requests"] = state["total_requests"]
    
    overall_avg_resp = state["total_request_time"] / state["total_requests"] if state["total_requests"] > 0 else 0
    
    median_db = statistics.median(state["retrieval_times"]) if state["retrieval_times"] else 0
    overall_avg_db = (state["total_db_time"] / state["total_db_calls"]) if state["total_db_calls"] > 0 else 0
    peak_db = state["max_db_latency"]
    
    avg_ttft = statistics.mean(state["ttft_times"]) if state["ttft_times"] else 0
    last_ttft = state["last_ttft"]
    
    percent = (state["total_requests"] / state["target_requests"]) * 100
    
    table = Table(show_header=False, box=None)
    table.add_row("Elapsed Time:", f"[cyan]{int(elapsed)}s[/cyan]")
    table.add_row("Query Mode:", f"[bold white]{state['query_mode']}[/bold white]")
    table.add_row("Progress:", f"[bold green]{percent:.1f}%[/bold green]")
    table.add_row("Requests/s:", f"[yellow]{rps}[/yellow]")
    table.add_row("Tokens/s:", f"[bold magenta]{tps}[/bold magenta]")
    table.add_row("Peak TPS:", f"[red]{state['max_tps']}[/red]")
    table.add_row("Last TTFT:", f"[bold white]{last_ttft*1000:.1f}ms[/bold white]")
    table.add_row("Avg TTFT:", f"[bold yellow]{avg_ttft*1000:.1f}ms[/bold yellow]")
    table.add_row("Overall Avg:", f"[bold cyan]{overall_avg_resp:.2f}s[/bold cyan]")
    table.add_row("DB Latency (Med):", f"[bold cyan]{median_db*1000:.1f}ms[/bold cyan]")
    
    if is_final:
        theoretical_sequential = state["total_request_time"]
        efficiency_gain = theoretical_sequential / elapsed if elapsed > 0 else 1
        
        # Format to min:sec
        seq_mins = int(theoretical_sequential // 60)
        seq_secs = int(theoretical_sequential % 60)
        seq_time_str = f"{seq_mins}m {seq_secs}s"
        
        table.add_row("-" * 20, "")
        table.add_row("Seq. Time:", f"[bold yellow]{seq_time_str}[/bold yellow]")
        table.add_row("Multiplier:", f"[bold green]{efficiency_gain:.2f}x Speedup[/bold green]")
        table.add_row("Avg DB (Total):", f"[bold blue]{overall_avg_db*1000:.1f}ms[/bold blue]")
    else:
        table.add_row("DB Peak (Switch):", f"[bold red]{peak_db*1000:.1f}ms[/bold red]")
    
    return Panel(table, title="[bold]Metrics[/bold]", border_style="bright_blue")

def get_responses_pane():
    content = Text.from_markup("\n".join(state["responses"]))
    return Panel(content, title=f"[bold]Live Answer Stream ({state['query_mode']})[/bold]", border_style="green")

def get_footer():
    gpu_text = f"GPU: [bold red]{state['gpu_util']}%[/bold red] | Temp: [bold orange1]{state['gpu_temp']}°C[/bold orange1] | VRAM: [bold yellow]{state['gpu_mem_used']}/{state['gpu_mem_total']}MB[/bold yellow]"
    qdrant_text = f"Qdrant: [bold cyan]{state['qdrant_points']} pts[/bold cyan] ([bold blue]{state['qdrant_vectors']} indexed[/bold blue])"
    
    content = Align.center(f"{gpu_text}  ┃  {qdrant_text}")
    return Panel(content, border_style="white")

async def main():
    console = Console()
    
    print("=== SHRED_V2.PY - NEXT GEN GRANITE RAG STRESS TEST ===")
    try:
        choice = input("Concurrent Sessions (1, 16, 64, 128, 256) [default 16]: ")
        state["num_sessions"] = int(choice) if choice else 16
        
        limit_choice = input("Total Requests to Shred [default 500]: ")
        state["target_requests"] = int(limit_choice) if limit_choice else 500
        
        print("\n[Mode Selector]")
        print("1. Similar Queries (Tech Only - Locality testing)")
        print("2. Disparate Queries (Tech + 1984 - Context Switching)")
        mode_choice = input("Choose Mode (1 or 2) [default 1]: ")
        
        if mode_choice == "2":
            state["active_queries"] = TECH_QUERIES + DYSTOPIAN_QUERIES
            state["query_mode"] = "Disparate"
        else:
            state["active_queries"] = TECH_QUERIES
            state["query_mode"] = "Similar"
            
    except ValueError:
        pass

    q_client = AsyncQdrantClient(host=QDRANT_URL, port=QDRANT_PORT)
    limits = httpx.Limits(max_keepalive_connections=state["num_sessions"], max_connections=state["num_sessions"] + 10)
    
    layout = make_layout()
    layout["header"].update(Panel(Align.center("[bold red]SHREDDER V2[/bold red] - Stressing Granite RAG Stack"), border_style="red"))
    
    async with httpx.AsyncClient(limits=limits) as http_client:
        state["start_time"] = time.time()
        
        gpu_task = asyncio.create_task(get_gpu_stats())
        qdrant_task = asyncio.create_task(get_qdrant_stats(q_client))
        
        workers = [
            asyncio.create_task(session_worker(i, q_client, http_client))
            for i in range(state["num_sessions"])
        ]
        
        with Live(layout, refresh_per_second=4, screen=True) as live:
            while state["total_requests"] < state["target_requests"]:
                layout["main"]["stats"].update(get_stats_table())
                layout["main"]["responses"].update(get_responses_pane())
                layout["footer"].update(get_footer())
                await asyncio.sleep(0.25)
            
            # Final update to indicate completion
            state["is_running"] = False
            layout["header"].update(Panel(Align.center(f"[bold green]SHREDDING COMPLETE ({state['query_mode']})[/bold green]"), border_style="green"))
            layout["main"]["stats"].update(get_stats_table(is_final=True))
            layout["main"]["responses"].update(get_responses_pane())
            layout["footer"].update(Panel(Align.center("[bold yellow]Demo Finished. Press ENTER to close TUI.[/bold yellow]"), border_style="yellow"))
            
            # Wait for user to acknowledge
            await asyncio.to_thread(input, "")
        
        for w in workers: w.cancel()
        gpu_task.cancel()
        qdrant_task.cancel()
        await q_client.close()

    total_time = time.time() - state["start_time"]
    theoretical_sequential = state["total_request_time"]
    efficiency_gain = theoretical_sequential / total_time if total_time > 0 else 1
    
    seq_mins = int(theoretical_sequential // 60)
    seq_secs = int(theoretical_sequential % 60)

    print(f"\n[COMPLETE] Mode: {state['query_mode']} | Time: {total_time:.2f}s")
    print(f"Final Average: {state['total_tokens'] / total_time:.2f} tokens/sec")
    print(f"Peak Performance: {state['max_tps']} tokens/sec")
    print(f"Theoretical Sequential Time (Ollama-style): {seq_mins}m {seq_secs}s")
    print(f"vLLM Concurrency Multiplier: {efficiency_gain:.2f}x faster than serial processing")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
