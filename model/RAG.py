from sentence_transformers import SentenceTransformer  # Import SentenceTransformer for embedding generation
import faiss  # Import FAISS for efficient similarity search
import os  # Import os module for file handling
import numpy as np  # Import numpy for saving/loading embeddings
from utility import converter
class RAGSystem:
    def __init__(self, 
                 embedding_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 
                 embedding_dim=384, 
                 data_dir='data/',
                 store_dir='store/',  # New folder for storing index and doc store
                 index_filename='faiss_index.idx',
                 doc_store_filename='documents.npy'):
        """Initializes the RAG (Retrieval-Augmented Generation) system.
        
        Args:
            embedding_model (str): Pre-trained model for generating embeddings.
            embedding_dim (int): Dimensionality of the embeddings.
            data_dir (str): Directory containing the document files.
            index_path (str): Path to save/load the FAISS index.
            doc_store_path (str): Path to save/load the document store.
        """
        self.retriever = SentenceTransformer(embedding_model)
        self.data_dir = data_dir
        self.store_dir = store_dir
        
        # Ensure the store directory exists
        if not os.path.exists(self.store_dir):
            os.makedirs(self.store_dir)
        
        self.index_path = os.path.join(self.store_dir, index_filename)
        self.doc_store_path = os.path.join(self.store_dir, doc_store_filename)
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []
        self.load_index()
    
    def load_index(self):
        """Loads the FAISS index and document store if available, otherwise loads documents."""
        if os.path.exists(self.index_path) and os.path.exists(self.doc_store_path):
            self.index = faiss.read_index(self.index_path)
            self.documents = np.load(self.doc_store_path, allow_pickle=True).tolist()
        else:
            self.load_documents()
    
    def save_index(self):
        """Saves the FAISS index and document store to disk."""
        faiss.write_index(self.index, self.index_path)
        np.save(self.doc_store_path, np.array(self.documents, dtype=object))

    def load_documents(self):
            """Loads and embeds documents from the data directory."""
            docs = []
            for file in os.listdir(self.data_dir):
                file_path = os.path.join(self.data_dir, file)
                if file.lower().endswith('.pdf'):
                    # Use python-docx to read DOCX files
                    pdf_text = converter.pdf_to_string(file_path)


                    docs.append(pdf_text)
                elif file.lower().endswith('.txt'):
                    # Fallback for reading plain text files
                    with open(file_path, 'r', encoding='utf-8') as f:
                        docs.append(f.read())
            self.add_documents(docs)
    def add_documents(self, docs):
        """Embeds and adds documents to the FAISS index.
        
        Args:
            docs (list of str): List of document texts to embed and store.
        """
        embeddings = self.retriever.encode(docs, convert_to_numpy=True)
        self.index.add(embeddings)
        self.documents.extend(docs)
        self.save_index()
    
    def retrieve(self, query, top_k=5):
        """Retrieves the most relevant documents for a given query.
        
        Args:
            query (str): Input query string.
            top_k (int): Number of top relevant documents to retrieve.
        
        Returns:
            list of tuples: Each tuple contains a document and its similarity score.
        """
        query_embedding = self.retriever.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        return [(self.documents[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
    
    def generate_prompt(self, query):
        """Retrieves relevant documents and generates a response prompt.
        
        Args:
            query (str): User input query.
        
        Returns:
            str: Prompt formatted for an LLM.
        """
        relevant_docs = self.retrieve(query)
        retrieved_context = "\n".join([doc[0] for doc in relevant_docs])
        prompt = f"請根據以下資料回答問題：\n\n{retrieved_context}\n\n問題：{query}\n\n回答："
        return prompt
