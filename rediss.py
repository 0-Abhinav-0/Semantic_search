import os
import redis
import numpy as np
import PyPDF2
import re
from sentence_transformers import SentenceTransformer

UPLOAD_FOLDER = "uploads"
REDIS_HOST = 'localhost'
REDIS_PORT = 6379

# Load a better model for semantic embeddings (higher accuracy)
model = SentenceTransformer('all-mpnet-base-v2')

# Connect to Redis
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# Clear existing Redis data to avoid conflicts with old format
print("Clearing existing Redis data...")
r.flushdb()
print("Redis database cleared.")

def clean_text(text):
    """Clean and normalize extracted text for better embedding quality"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers and common PDF artifacts
    text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Fix common OCR issues
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    return text.strip()

def chunk_text(text, chunk_size=400, overlap=100):
    """Split text into overlapping chunks for better semantic search"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)
        
        if current_word_count + sentence_word_count > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            
            # Keep overlap for context
            overlap_words = []
            overlap_count = 0
            for prev_sentence in reversed(current_chunk):
                prev_words = prev_sentence.split()
                if overlap_count + len(prev_words) <= overlap:
                    overlap_words.insert(0, prev_sentence)
                    overlap_count += len(prev_words)
                else:
                    break
            
            current_chunk = overlap_words
            current_word_count = overlap_count
        
        current_chunk.append(sentence)
        current_word_count += sentence_word_count
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# ----------------- Process PDFs -----------------
print("Processing PDFs with improved chunking and embeddings...")
for filename in os.listdir(UPLOAD_FOLDER):
    if filename.endswith(".pdf"):
        path = os.path.join(UPLOAD_FOLDER, filename)
        print(f"Processing: {filename}")
        
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "".join([page.extract_text() or "" for page in reader.pages])
        
        # Clean the text
        text = clean_text(text)
        
        # Split into chunks for better semantic matching
        chunks = chunk_text(text)
        
        for i, chunk in enumerate(chunks):
            # Generate embedding for each chunk (normalized)
            vector = model.encode(chunk, convert_to_numpy=True, normalize_embeddings=True)
            
            # Store in Redis with chunk information
            chunk_key = f"chunk:{filename}::{i+1}"
            r.set(chunk_key, vector.tobytes())
            r.set(f"text:{filename}::{i+1}", chunk)  # Store text for display
        
        print(f"Stored {len(chunks)} chunks for {filename}")

# ----------------- Semantic Search -----------------
def find_most_similar_content(query, top_k=2):
    """Find the most semantically similar content chunks"""
    # Create normalized query embedding
    query_vec = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    
    results = []
    # Search through all chunks
    for key in r.keys("chunk:*"):
        chunk_vec = np.frombuffer(r.get(key), dtype='float32')
        
        # Calculate cosine similarity (both vectors are normalized)
        similarity = np.dot(query_vec, chunk_vec)
        
        # Extract filename and chunk info from key
        key_str = key.decode()
        parts = key_str.replace("chunk:", "").split("::")
        filename = parts[0]
        chunk_num = parts[1]
        
        # Get the actual text content
        text_key = f"text:{filename}::{chunk_num}"
        text_content = r.get(text_key)
        if text_content:
            text_content = text_content.decode('utf-8')
        else:
            text_content = "Text not available"
        
        results.append({
            'filename': filename,
            'chunk': chunk_num,
            'similarity': similarity,
            'text': text_content
        })
    
    # Sort by similarity (highest first) and return top results
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]

# ----------------- User Query -----------------
print("\n🔍 Enhanced Semantic Search Ready!")
print("Try queries like: 'graduate associate engineer', 'software developer', 'project manager', etc.\n")

while True:
    query = input("Enter your search query (or 'exit'): ")
    if query.lower() in ['exit', 'quit']:
        break
    
    # Find the most similar content chunks
    top_results = find_most_similar_content(query, top_k=2)
    
    if not top_results:
        print("No results found. Try a different query.")
        continue
    
    print(f"\n🎯 Top {len(top_results)} most similar results for '{query}':")
    print("=" * 80)
    
    for i, result in enumerate(top_results, 1):
        print(f"\n📄 Result {i}: {result['filename']} (chunk {result['chunk']})")
        print(f"🔥 Similarity Score: {result['similarity']:.4f}")
        print(f"📝 Content Preview:")
        # Show first 300 characters of the content
        preview = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
        print(f"   {preview}")
        print("-" * 80)
    
    print()
