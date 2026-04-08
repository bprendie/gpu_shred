import requests
import uuid
import glob
import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

# --- Configuration ---
# Pointing to your vllm_embed container on port 8001
EMBED_URL = "http://localhost:8001/v1/embeddings"
EMBED_MODEL = "ibm-granite/granite-embedding-125m-english"
QDRANT_URL = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "granite_pdf_test"

# Lowered to 400 to prevent the 512-token overflow on dense technical pages
CHUNK_SIZE = 400  
CHUNK_OVERLAP = 50 

client = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)

def get_embeddings(texts, batch_size=16):
    """Fetch embeddings in batches and handle token-limit errors gracefully."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = {
            "model": EMBED_MODEL,
            "input": batch
        }
        try:
            res = requests.post(EMBED_URL, json=payload, timeout=30)
            if res.status_code == 200:
                batch_vectors = [data["embedding"] for data in res.json()["data"]]
                all_embeddings.extend(batch_vectors)
            else:
                print(f"\n[Error] Batch {i//batch_size} failed: {res.text}")
                # We return a partial list; the main loop will sync vectors to chunks
        except Exception as e:
            print(f"\n[Exception] Connection error during embedding: {e}")
            
    return all_embeddings

def setup_qdrant():
    """Wipes and recreates the collection with the correct 768d vector size."""
    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

def main():
    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        print("No PDF files found in current directory.")
        return
        
    setup_qdrant()
    
    for pdf_file in pdf_files:
        print(f"\n--- Ingesting: {pdf_file} ---")
        doc = fitz.open(pdf_file)
        
        # Process page by page to capture metadata
        for page_num, page in enumerate(tqdm(doc, desc="Processing Pages")):
            page_text = page.get_text().strip()
            if not page_text:
                continue
                
            # Create chunks for this specific page
            chunks = [page_text[i:i+CHUNK_SIZE] for i in range(0, len(page_text), CHUNK_SIZE - CHUNK_OVERLAP)]
            
            # Get embeddings from the GPU container
            vectors = get_embeddings(chunks)
            
            # CRITICAL: Only upsert if we actually generated vectors
            # This avoids the "Empty update request" 400 error from Qdrant
            if vectors and len(vectors) == len(chunks):
                points = [
                    PointStruct(
                        id=str(uuid.uuid4()), 
                        vector=v, 
                        payload={
                            "text": c, 
                            "source": pdf_file, 
                            "page": page_num + 1
                        }
                    )
                    for v, c in zip(vectors, chunks)
                ]
                client.upsert(collection_name=COLLECTION_NAME, points=points)
            elif vectors:
                print(f"\n[Warning] Mismatch on Page {page_num+1}: {len(vectors)} vectors for {len(chunks)} chunks. Skipping.")

    print("\n=== Ingestion Complete with Page Metadata ===")

if __name__ == "__main__":
    main()
