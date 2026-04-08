import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# --- Configuration ---
CHAT_URL = "http://localhost:8000/v1/chat/completions"
# Updated to Micro as discussed for VRAM efficiency
MODEL_CHAT = "ibm-granite/granite-4.0-h-micro" 
MODEL_EMBED_PATH = "ibm-granite/granite-embedding-125m-english"
QDRANT_URL = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "granite_pdf_test"

# Initialize Qdrant Client
client = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)

# Initialize Embedding Model on CPU
print(f"Loading embedding model [{MODEL_EMBED_PATH}] to CPU...")
embed_model = SentenceTransformer(MODEL_EMBED_PATH, device="cpu")

def get_query_vector(text):
    """Convert the user question into a vector using local CPU."""
    return embed_model.encode(text).tolist()

def retrieve_context(query_vector, limit=3):
    """Search Qdrant and extract text, source, and page metadata."""
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True
    ).points
    
    context_blocks = []
    citations = []
    
    for hit in search_result:
        text = hit.payload.get("text", "")
        source = hit.payload.get("source", "Unknown")
        page = hit.payload.get("page", "N/A") # Expecting 'page' field
        
        # Label the context blocks for the LLM
        context_blocks.append(f"[File: {source}, Page: {page}]\n{text}")
        citations.append(f"{source} (p. {page})")
    
    context_str = "\n---\n".join(context_blocks)
    return context_str, list(set(citations))

def chat_with_vllm(prompt, context):
    """Send the augmented prompt with citation instructions."""
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
        "temperature": 0.0 # Accuracy over creativity
    }
    
    res = requests.post(CHAT_URL, json=payload)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

def main():
    print(f"\nRAG Chat Online | Context: {COLLECTION_NAME}")
    print(f"Generative Model: {MODEL_CHAT}")
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("USER: ")
        if query.lower() in ["exit", "quit"]:
            break
            
        try:
            # 1. Vectorize
            vector = get_query_vector(query)
            
            # 2. Retrieve with Metadata
            context, citations = retrieve_context(vector)
            
            # 3. Generate Answer
            answer = chat_with_vllm(query, context)
            
            print(f"\nASSISTANT: {answer}")
            print(f"\n[Verified Sources: {', '.join(citations)}]\n")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
