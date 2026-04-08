import requests
import time
import json
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# --- Configuration ---
CHAT_URL = "http://localhost:8000/v1/chat/completions"
MODEL_CHAT = "ibm-granite/granite-4.0-h-micro" 
MODEL_EMBED_PATH = "ibm-granite/granite-embedding-125m-english"
QDRANT_URL = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "granite_pdf_test"

client = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)

print(f"Loading embedding model [{MODEL_EMBED_PATH}] to CPU...")
embed_model = SentenceTransformer(MODEL_EMBED_PATH, device="cpu")

def get_query_vector(text):
    return embed_model.encode(text).tolist()

def retrieve_context(query_vector, limit=3):
    start_db = time.perf_counter()
    
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True
    ).points
    
    db_latency_ms = (time.perf_counter() - start_db) * 1000
    
    context_blocks = []
    citations = []
    
    for hit in search_result:
        text = hit.payload.get("text", "")
        source = hit.payload.get("source", "Unknown")
        page = hit.payload.get("page", "N/A")
        
        context_blocks.append(f"[File: {source}, Page: {page}]\n{text}")
        citations.append(f"{source} (p. {page})")
    
    context_str = "\n---\n".join(context_blocks)
    return context_str, list(set(citations)), db_latency_ms

def chat_with_vllm_stream(prompt, context):
    system_prompt = (
        "You are a technical assistant. Use the provided context to answer the user's question. "
        "The context is divided into blocks labeled with File name and Page number. "
        "You MUST cite the File and Page number for every fact you state in your answer. "
        "If you cannot find the answer, say you don't know."
        f"\n\nCONTEXT:\n{context}"
    )
    
    payload = {
        "model": MODEL_CHAT,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True} # Crucial for getting accurate token counts from vLLM
    }
    
    start_req = time.perf_counter()
    ttft = None
    full_content = ""
    tokens = 0
    
    print(f"\nASSISTANT: ", end="", flush=True)
    
    # Open the connection and iterate over the stream
    with requests.post(CHAT_URL, json=payload, stream=True) as res:
        res.raise_for_status()
        for line in res.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    
                    if data_str == "[DONE]":
                        break
                        
                    try:
                        chunk = json.loads(data_str)
                        
                        # Trap TTFT the moment the first string of content arrives
                        if ttft is None and chunk.get("choices") and chunk["choices"][0]["delta"].get("content"):
                            ttft = time.perf_counter() - start_req
                            
                        # Stream the text to the console
                        if chunk.get("choices"):
                            content = chunk["choices"][0]["delta"].get("content", "")
                            full_content += content
                            print(content, end="", flush=True)
                            
                        # vLLM sends usage data in the final chunk if stream_options requested it
                        if chunk.get("usage"):
                            tokens = chunk["usage"].get("completion_tokens", 0)
                            
                    except json.JSONDecodeError:
                        continue
                        
    end_req = time.perf_counter()
    print() # Cap the line after the stream finishes
    
    if tokens == 0:
        tokens = len(full_content.split()) # Fallback if usage is missing
        
    total_time = end_req - start_req
    ttft = ttft if ttft else total_time
    gen_time = total_time - ttft
    
    # Isolate pure decoding speed
    tok_s = tokens / gen_time if gen_time > 0 else 0.0
    
    return ttft, tok_s, tokens

def main():
    print(f"\nRAG Chat Online | Context: {COLLECTION_NAME}")
    print(f"Generative Model: {MODEL_CHAT}")
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("USER: ")
        if query.lower() in ["exit", "quit"]:
            break
            
        try:
            vector = get_query_vector(query)
            context, citations, db_latency = retrieve_context(vector)
            
            # The generation now prints to stdout automatically during this call
            ttft, tok_s, total_tokens = chat_with_vllm_stream(query, context)
            
            print(f"\n[Verified Sources: {', '.join(citations)}]")
            print(f"[Metrics: DB Latency: {db_latency:.2f}ms | TTFT: {ttft*1000:.2f}ms | Output Tokens: {total_tokens} | Gen Speed: {tok_s:.2f} tok/s]\n")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
