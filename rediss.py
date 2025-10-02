import os
import redis
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer

# ------------------- CONFIG -------------------
UPLOAD_FOLDER = "uploads"  # folder with PDFs
REDIS_HOST = 'localhost'
REDIS_PORT = 6379

# Load Hugging Face embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
VECTOR_DIM = model.get_sentence_embedding_dimension()

# Connect to Redis
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# ------------------- CLEAR OLD PDF KEYS -------------------
for key in r.keys("pdf:*"):
    r.delete(key)
print("Cleared old PDF keys.")

# ------------------- EMBEDDING FUNCTION -------------------
def embed_text(text: str) -> np.ndarray:
    """Generate semantic embedding for text using Hugging Face."""
    return model.encode(text, convert_to_numpy=True)

# ------------------- READ PDFs AND STORE IN REDIS -------------------
for filename in os.listdir(UPLOAD_FOLDER):
    if filename.endswith(".pdf"):
        path = os.path.join(UPLOAD_FOLDER, filename)
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        
        vector = embed_text(text)
        key = f"pdf:{filename}"
        r.hset(key, mapping={
            "filename": filename,
            "vector": vector.tobytes()
        })
        print(f"Stored {filename} in Redis.")

# ------------------- SEMANTIC SEARCH FUNCTIONS -------------------
def get_all_vectors():
    vectors = []
    for key in r.keys("pdf:*"):
        if r.type(key).decode() != "hash":
            continue
        data = r.hget(key, "vector")
        if data:
            vectors.append((key.decode(), np.frombuffer(data, dtype='float32')))
    return vectors

def find_most_similar(query_text, top_k=3):
    query_vector = embed_text(query_text)
    vectors = get_all_vectors()
    if not vectors:
        return []
    
    # Cosine similarity
    sims = [(k, np.dot(query_vector, v) / (np.linalg.norm(query_vector) * np.linalg.norm(v))) for k, v in vectors]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]

# ------------------- QUERY LOOP -------------------
print("\nSemantic search ready! Type 'exit' to quit.\n")

while True:
    query = input("Enter your search query: ")
    if query.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    results = find_most_similar(query, top_k=3)
    if not results:
        print("No PDFs found in Redis.\n")
        continue

    print("\nTop similar PDFs:")
    for key, score in results:
        filename = r.hget(key, "filename").decode()
        print(f"{filename} (similarity: {score:.4f})")
    print()
