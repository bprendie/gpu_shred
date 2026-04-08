import requests
import uuid
import glob
import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
from transformers import AutoTokenizer

# --- Configuration ---
EMBED_URL = "http://localhost:8001/v1/embeddings"
EMBED_MODEL = "ibm-granite/granite-embedding-125m-english"
QDRANT_URL = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "granite_pdf_test" # Kept per your requirement

# Hard limit for Granite-125m is 512; we use 500 for safety.
MAX_TOKENS = 500  
OVERLAP_TOKENS = 50 

# Load tokenizer (ensure .venv is active or transformers is installed)
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
client = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)

def get_embeddings(texts, batch_size=16):
    """Fetch 768d vectors from the vLLM container."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = {"model": EMBED_MODEL, "input": batch}
        try:
            res = requests.post(EMBED_URL, json=payload, timeout=60)
            if res.status_code == 200:
                batch_vectors = [data["embedding"] for data in res.json()["data"]]
                all_embeddings.extend(batch_vectors)
            else:
                print(f"\n[Error] Embedding failed: {res.text}")
        except Exception as e:
            print(f"\n[Exception] Connection error: {e}")
    return all_embeddings

def setup_qdrant():
    """Create/Wipe the collection with standard 768d settings."""
    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

def main():
    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        print("No PDF files found.")
        return
        
    setup_qdrant()
    
    for pdf_file in pdf_files:
        print(f"\n--- Processing: {pdf_file} ---")
        doc = fitz.open(pdf_file)
        
        full_tokens = []
        page_boundaries = [] 

        # 1. Map tokens to page numbers to preserve references
        for page_num, page in enumerate(doc):
            page_text = page.get_text() + "\n"
            # Encode without special tokens to get a clean stream
            page_tokens = tokenizer.encode(page_text, add_special_tokens=False)
            
            start_index = len(full_tokens)
            full_tokens.extend(page_tokens)
            end_index = len(full_tokens)
            
            page_boundaries.append({
                "page": page_num + 1,
                "start": start_index,
                "end": end_index
            })

        # 2. Slice the token stream and map back to metadata
        points = []
        # Step through the token list with overlap
        for i in range(0, len(full_tokens), MAX_TOKENS - OVERLAP_TOKENS):
            chunk_tokens = full_tokens[i : i + MAX_TOKENS]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # Identify the page number by checking where the chunk starts
            chunk_page = 1
            for boundary in page_boundaries:
                if i >= boundary["start"] and i < boundary["end"]:
                    chunk_page = boundary["page"]
                    break

            # 3. Get 768d Embedding
            vectors = get_embeddings([chunk_text])
            if not vectors:
                continue
                
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()), 
                    vector=vectors[0], 
                    payload={
                        "text": chunk_text, 
                        "source": pdf_file,
                        "page": chunk_page 
                    }
                )
            )
            
            # Break if we've processed the end of the file
            if i + MAX_TOKENS >= len(full_tokens):
                break
        
        # 4. Batch upsert to Qdrant
        if points:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"Ingested {len(points)} high-context chunks (768d).")

    print("\n=== Ingestion Complete: Standardized 768d Metadata Space ===")

if __name__ == "__main__":
    main()
