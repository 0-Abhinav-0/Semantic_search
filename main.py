import os
import re
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from collections import Counter
from typing import List, Dict, Tuple

class PDFSemanticSearch:
    def __init__(self, pdf_folder_path, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the PDF Semantic Search system
        
        Args:
            pdf_folder_path (str): Path to folder containing PDF files
            model_name (str): Name of the sentence transformer model
                             Options for better accuracy:
                             - 'all-MiniLM-L6-v2' (default, fast, 384 dim)
                             - 'all-mpnet-base-v2' (better accuracy, 768 dim)
                             - 'multi-qa-mpnet-base-dot-v1' (optimized for Q&A)
        """
        self.pdf_folder_path = pdf_folder_path
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = []
        self.pdf_names = []
        self.metadata = []  # Store additional metadata per chunk
        
    def clean_text(self, text):
        """Clean and normalize extracted text for better embedding quality"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (common pattern)
        text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
        
        # Remove common PDF artifacts
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Fix common OCR issues
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        
        return text.strip()
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a single PDF file with better handling"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            # Clean the extracted text
            text = self.clean_text(text)
            return text
            
        except Exception as e:
            print(f"Error reading {pdf_path}: {str(e)}")
            return ""
    
    def chunk_text_smart(self, text, chunk_size=400, overlap=100):
        """
        Smart text chunking that respects sentence boundaries
        
        Args:
            text (str): Text to chunk
            chunk_size (int): Target chunk size in words
            overlap (int): Overlap between chunks in words
        """
        # Split into sentences first (more semantic coherence)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_word_count + sentence_word_count > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Keep overlap sentences for context
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
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def get_base_filename(self, pdf_name):
        """Extract base filename from 'filename.pdf (chunk N)' format"""
        return pdf_name.split(' (chunk ')[0]
    
    def index_pdfs(self, save_index=True, index_file="pdf_index.pkl", incremental=True):
        """
        Extract text from all PDFs and create embeddings with metadata
        """
        print("Starting PDF indexing...")
        
        pdf_files = [f for f in os.listdir(self.pdf_folder_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print("No PDF files found in the specified folder!")
            return
        
        # Handle incremental indexing
        if incremental and self.documents:
            existing_files = set()
            for pdf_name in self.pdf_names:
                base_file = self.get_base_filename(pdf_name)
                existing_files.add(base_file)
            
            new_files = [f for f in pdf_files if f not in existing_files]
            
            if not new_files:
                print("No new PDF files found. Index is up to date.")
                return
            
            print(f"Found {len(new_files)} new files to index: {new_files}")
            files_to_process = new_files
        else:
            print("Performing full reindex...")
            self.documents = []
            self.embeddings = []
            self.pdf_names = []
            self.metadata = []
            files_to_process = pdf_files
        
        # Process files
        new_documents = []
        new_pdf_names = []
        new_metadata = []
        
        for pdf_file in files_to_process:
            pdf_path = os.path.join(self.pdf_folder_path, pdf_file)
            print(f"Processing: {pdf_file}")
            
            text = self.extract_text_from_pdf(pdf_path)
            
            if text:
                # Use smart chunking for better semantic coherence
                chunks = self.chunk_text_smart(text)
                
                for i, chunk in enumerate(chunks):
                    new_documents.append(chunk)
                    new_pdf_names.append(f"{pdf_file} (chunk {i+1}/{len(chunks)})")
                    new_metadata.append({
                        'file': pdf_file,
                        'chunk_id': i + 1,
                        'total_chunks': len(chunks),
                        'word_count': len(chunk.split())
                    })
        
        if not new_documents:
            print("No text extracted from new PDFs!")
            return
        
        print(f"Creating embeddings for {len(new_documents)} new text chunks...")
        
        # Create embeddings with normalization for better similarity scoring
        new_embeddings = self.model.encode(
            new_documents, 
            show_progress_bar=True,
            normalize_embeddings=True  # Normalizes for better cosine similarity
        )
        
        # Add new data
        self.documents.extend(new_documents)
        self.pdf_names.extend(new_pdf_names)
        self.metadata.extend(new_metadata)
        
        if len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        if save_index:
            self.save_index(index_file)
        
        print(f"Indexing complete! Total documents: {len(self.documents)}")
    
    def save_index(self, filename="pdf_index.pkl"):
        """Save the index to disk"""
        index_data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'pdf_names': self.pdf_names,
            'metadata': self.metadata
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"Index saved to {filename}")
    
    def load_index(self, filename="pdf_index.pkl"):
        """Load a previously saved index"""
        try:
            with open(filename, 'rb') as f:
                index_data = pickle.load(f)
            
            self.documents = index_data['documents']
            self.embeddings = index_data['embeddings']
            self.pdf_names = index_data['pdf_names']
            self.metadata = index_data.get('metadata', [])
            
            print(f"Index loaded from {filename}")
            print(f"Loaded {len(self.documents)} documents")
            
        except FileNotFoundError:
            print(f"Index file {filename} not found. Please run index_pdfs() first.")
        except Exception as e:
            print(f"Error loading index: {str(e)}")
    
    def search(self, query, top_k=5, similarity_threshold=0.2, 
               use_query_expansion=True, rerank=True):
        """
        Enhanced semantic search with query expansion and reranking
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            similarity_threshold (float): Minimum similarity score (0.2 is better for normalized embeddings)
            use_query_expansion (bool): Expand query with variations
            rerank (bool): Apply reranking for better results
        """
        if not self.documents:
            print("No documents indexed. Please run index_pdfs() first.")
            return []
        
        # Query expansion for better recall
        queries_to_search = [query]
        if use_query_expansion:
            # Add cleaned query
            cleaned = self.clean_text(query)
            if cleaned != query:
                queries_to_search.append(cleaned)
            
            # Add query with question words removed (for declarative matching)
            question_removed = re.sub(r'\b(what|who|where|when|why|how|is|are|does|do)\b', 
                                     '', query, flags=re.IGNORECASE).strip()
            if question_removed and question_removed != query:
                queries_to_search.append(question_removed)
        
        # Encode all query variations
        query_embeddings = self.model.encode(
            queries_to_search,
            normalize_embeddings=True
        )
        
        # Use max similarity across all query variations (better recall)
        all_similarities = cosine_similarity(query_embeddings, self.embeddings)
        similarities = np.max(all_similarities, axis=0)
        
        # Get more candidates initially for reranking
        candidate_count = min(top_k * 5, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:candidate_count]
        
        # Filter by threshold
        valid_indices = [idx for idx in top_indices if similarities[idx] >= similarity_threshold]
        
        if not valid_indices:
            return []
        
        # Count file occurrences for relevance boosting
        base_filenames = [self.get_base_filename(self.pdf_names[i]) for i in valid_indices]
        filename_counts = Counter(base_filenames)
        most_common_file = filename_counts.most_common(1)[0][0]
        
        # Build results with enhanced scoring
        results_with_priority = []
        
        for idx in valid_indices:
            base_filename = self.get_base_filename(self.pdf_names[idx])
            is_most_common = (base_filename == most_common_file)
            file_count = filename_counts[base_filename]
            
            # Enhanced scoring combining similarity and file frequency
            base_score = float(similarities[idx])
            
            # Boost score for chunks from the most relevant file
            if is_most_common and file_count > 1:
                # Boost by up to 10% based on how many chunks matched
                frequency_boost = min(0.1, (file_count / len(valid_indices)) * 0.2)
                enhanced_score = base_score * (1 + frequency_boost)
            else:
                enhanced_score = base_score
            
            result = {
                'pdf_name': self.pdf_names[idx],
                'text': self.documents[idx],
                'score': base_score,
                'enhanced_score': enhanced_score,
                'is_most_common': is_most_common,
                'file_count': file_count,
                'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
            }
            results_with_priority.append(result)
        
        # Sort by enhanced score
        results_with_priority.sort(key=lambda x: -x['enhanced_score'])
        
        # Optional: Rerank top results using cross-encoder (more accurate but slower)
        if rerank and len(results_with_priority) > 1:
            results_with_priority = self._rerank_results(query, results_with_priority[:top_k * 2])
        
        # Return final results
        final_results = []
        for result in results_with_priority[:top_k]:
            final_results.append({
                'pdf_name': result['pdf_name'],
                'text': result['text'],
                'score': result['score'],
                'metadata': result['metadata']
            })
        
        return final_results
    
    def _rerank_results(self, query, results):
        """
        Simple reranking based on keyword matching and length
        (Can be replaced with cross-encoder for even better accuracy)
        """
        query_terms = set(query.lower().split())
        
        for result in results:
            text_lower = result['text'].lower()
            
            # Count exact keyword matches
            keyword_matches = sum(1 for term in query_terms if term in text_lower)
            
            # Prefer chunks with more keyword matches
            keyword_boost = keyword_matches * 0.02
            
            # Slight preference for longer, more informative chunks
            length_score = min(0.03, len(result['text'].split()) / 1000)
            
            result['enhanced_score'] += keyword_boost + length_score
        
        results.sort(key=lambda x: -x['enhanced_score'])
        return results
    
    def search_multi_query(self, queries: List[str], top_k=5) -> Dict[str, List]:
        """
        Search multiple queries and return aggregated results
        Useful for complex information needs
        """
        all_results = {}
        for query in queries:
            results = self.search(query, top_k=top_k)
            all_results[query] = results
        return all_results
    
    def print_search_results(self, results):
        """Pretty print search results with enhanced information"""
        if not results:
            print("No results found.")
            return
        
        print(f"\nFound {len(results)} relevant results:\n")
        
        # Group results by base filename
        base_filenames = [self.get_base_filename(result['pdf_name']) for result in results]
        filename_counts = Counter(base_filenames)
        
        if filename_counts:
            most_common_file = filename_counts.most_common(1)[0]
            most_common_name, most_common_count = most_common_file
            
            if most_common_count > 1:
                print(f"🏆 Most Relevant File: {most_common_name} ({most_common_count} matches)")
                print("=" * 80)
            else:
                print("=" * 80)
        
        for i, result in enumerate(results, 1):
            base_file = self.get_base_filename(result['pdf_name'])
            is_most_common = (base_file == filename_counts.most_common(1)[0][0]) and filename_counts.most_common(1)[0][1] > 1
            
            print(f"\nResult {i}:" + (" ⭐ (Top File)" if is_most_common else ""))
            print(f"PDF: {result['pdf_name']}")
            print(f"Similarity Score: {result['score']:.4f}")
            
            # Show metadata if available
            if result.get('metadata'):
                meta = result['metadata']
                print(f"Chunk: {meta.get('chunk_id', '?')}/{meta.get('total_chunks', '?')} | Words: {meta.get('word_count', '?')}")
            
            # Smart preview that tries to show query-relevant parts
            preview = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
            print(f"Text Preview: {preview}")
            print("-" * 80)

    def add_new_pdfs(self):
        """Convenient method to add only new PDFs to existing index"""
        if os.path.exists("pdf_index.pkl"):
            self.load_index("pdf_index.pkl")
        self.index_pdfs(incremental=True)

# Example usage
def main():
    # Initialize with better model for improved accuracy
    # Options: 'all-MiniLM-L6-v2' (fast), 'all-mpnet-base-v2' (better), 'multi-qa-mpnet-base-dot-v1' (best for Q&A)
    pdf_folder = "uploads"
    
    # Use the more accurate model
    search_system = PDFSemanticSearch(pdf_folder, model_name='all-mpnet-base-v2')
    
    # Full reindex
    search_system.index_pdfs(incremental=False)
    
    # Or add new files
    # search_system.add_new_pdfs()
    
    # Or load existing
    # search_system.load_index("pdf_index.pkl")
    
    # Perform searches
    while True:
        query = input("\nEnter your search query (or 'quit' to exit): ")
        
        if query.lower() == 'quit':
            break
        
        # Enhanced search with all improvements
        results = search_system.search(
            query, 
            top_k=5,
            similarity_threshold=0.2,
            use_query_expansion=True,
            rerank=True
        )
        search_system.print_search_results(results)

if __name__ == "__main__":
    main()