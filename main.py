import os
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class PDFSemanticSearch:
    def __init__(self, pdf_folder_path, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the PDF Semantic Search system
        
        Args:
            pdf_folder_path (str): Path to folder containing PDF files
            model_name (str): Name of the sentence transformer model
        """
        self.pdf_folder_path = pdf_folder_path
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = []
        self.pdf_names = []
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a single PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error reading {pdf_path}: {str(e)}")
            return ""
    
    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Split text into overlapping chunks for better semantic search"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def index_pdfs(self, save_index=True, index_file="pdf_index.pkl", incremental=True):
        """
        Extract text from all PDFs and create embeddings
        
        Args:
            save_index (bool): Whether to save the index to disk
            index_file (str): Filename for saving the index
            incremental (bool): Whether to add only new files or reindex everything
        """
        print("Starting PDF indexing...")
        
        # Get all PDF files
        pdf_files = [f for f in os.listdir(self.pdf_folder_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print("No PDF files found in the specified folder!")
            return
        
        # If incremental and we have existing data, find new files
        if incremental and self.documents:
            existing_files = set()
            for pdf_name in self.pdf_names:
                # Extract base filename from "filename.pdf (chunk N)" format
                base_file = pdf_name.split(' (chunk ')[0]
                existing_files.add(base_file)
            
            new_files = [f for f in pdf_files if f not in existing_files]
            
            if not new_files:
                print("No new PDF files found. Index is up to date.")
                return
            
            print(f"Found {len(new_files)} new files to index: {new_files}")
            files_to_process = new_files
        else:
            print("Performing full reindex...")
            # Clear existing data for full reindex
            self.documents = []
            self.embeddings = []
            self.pdf_names = []
            files_to_process = pdf_files
        
        # Process files
        new_documents = []
        new_pdf_names = []
        
        for pdf_file in files_to_process:
            pdf_path = os.path.join(self.pdf_folder_path, pdf_file)
            print(f"Processing: {pdf_file}")
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            
            if text:
                # Split into chunks for better semantic search
                chunks = self.chunk_text(text)
                
                for i, chunk in enumerate(chunks):
                    new_documents.append(chunk)
                    new_pdf_names.append(f"{pdf_file} (chunk {i+1})")
        
        if not new_documents:
            print("No text extracted from new PDFs!")
            return
        
        print(f"Creating embeddings for {len(new_documents)} new text chunks...")
        
        # Create embeddings for new documents
        new_embeddings = self.model.encode(new_documents, show_progress_bar=True)
        
        # Add new data to existing data
        self.documents.extend(new_documents)
        self.pdf_names.extend(new_pdf_names)
        
        if len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Save index if requested
        if save_index:
            self.save_index(index_file)
        
        print(f"Indexing complete! Total documents: {len(self.documents)}")
    
    def save_index(self, filename="pdf_index.pkl"):
        """Save the index to disk"""
        index_data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'pdf_names': self.pdf_names
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
            
            print(f"Index loaded from {filename}")
            print(f"Loaded {len(self.documents)} documents")
            
        except FileNotFoundError:
            print(f"Index file {filename} not found. Please run index_pdfs() first.")
        except Exception as e:
            print(f"Error loading index: {str(e)}")
    
    def search(self, query, top_k=5, similarity_threshold=0.1):
        """
        Perform semantic search on indexed PDFs
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            similarity_threshold (float): Minimum similarity score
        
        Returns:
            list: List of tuples (pdf_name, text_chunk, similarity_score)
        """
        if not self.documents:
            print("No documents indexed. Please run index_pdfs() first.")
            return []
        
        # Create query embedding
        query_embedding = self.model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score >= similarity_threshold:
                results.append({
                    'pdf_name': self.pdf_names[idx],
                    'text': self.documents[idx],
                    'score': float(score)
                })
        
        return results
    
    def print_search_results(self, results):
        """Pretty print search results"""
        if not results:
            print("No results found.")
            return
        
        print(f"\nFound {len(results)} relevant results:\n")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"PDF: {result['pdf_name']}")
            print(f"Similarity Score: {result['score']:.4f}")
            print(f"Text Preview: {result['text'][:200]}...")
            print("-" * 80)

    def add_new_pdfs(self):
        """Convenient method to add only new PDFs to existing index"""
        # First load existing index if available
        if os.path.exists("pdf_index.pkl"):
            self.load_index("pdf_index.pkl")
        
        # Then add new files incrementally
        self.index_pdfs(incremental=True)

# Example usage
def main():
    # Initialize the search system - PDFs in 'uploads' folder
    pdf_folder = "uploads"  # Relative path to uploads folder
    # Or use absolute path: pdf_folder = os.path.join(os.getcwd(), "uploads")
    search_system = PDFSemanticSearch(pdf_folder)
    
    # Option 1: Full reindex (first time or when you want to rebuild everything)
    search_system.index_pdfs(incremental=False)
    
    # Option 2: Add only new files to existing index
    # search_system.add_new_pdfs()
    
    # Option 3: Or load existing index
    # search_system.load_index("pdf_index.pkl")
    
    # Perform searches
    while True:
        query = input("\nEnter your search query (or 'quit' to exit): ")
        
        if query.lower() == 'quit':
            break
            
        results = search_system.search(query, top_k=5)
        search_system.print_search_results(results)

if __name__ == "__main__":
    main()