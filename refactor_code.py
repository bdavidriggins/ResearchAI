import os
import json
import requests
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class RAGSystem:
    def __init__(self,
                 faiss_index_path,
                 faiss_metadata_path,
                 embedding_model_name='multi-qa-mpnet-base-dot-v1',
                 reranker_model_name='cross-encoder/ms-marco-TinyBERT-L-6',
                 ollama_api_url='http://localhost:11434/api/generate',
                 ollama_model="llama3.1:70b"):
        """
        Initialize the Retrieval-Augmented Generation (RAG) system.
        """
        # Load FAISS index and metadata
        self.index = faiss.read_index(faiss_index_path)
        with open(faiss_metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize reranker model
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
        self.reranker_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reranker_model.to(self.reranker_device)
        
        # Initialize Ollama client parameters
        self.ollama_api_url = ollama_api_url
        self.ollama_model = ollama_model

    def retrieve(self, query, top_k=100):
        """
        Retrieve top_k documents from the FAISS index based on the query.
        """
        query_embedding = self.embedding_model.encode(query).astype('float32')
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return [self.metadata[idx] for idx in indices[0]]

    def rerank(self, query, candidates, top_k=5):
        """
        Rerank the retrieved documents using a cross-encoder model.
        """
        inputs = [f"{query} [SEP] {candidate['text']}" for candidate in candidates]
        encoded = self.reranker_tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
        encoded = {k: v.to(self.reranker_device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.reranker_model(**encoded)
            scores = outputs.logits.squeeze().cpu().numpy()
        
        for i, candidate in enumerate(candidates):
            candidate['score'] = scores[i]
        
        sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        return sorted_candidates[:top_k]

    def generate(self, prompt, context):
        """
        Generate a response using the Ollama API.
        """
        full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
        payload = {
            "model": self.ollama_model,
            "prompt": full_prompt,
            "max_tokens": 4096,
            "temperature": 0.7,
            "stream": False
        }
        response = requests.post(self.ollama_api_url, json=payload)
        response.raise_for_status()
        return response.json().get('response', '')

    def rag_pipeline(self, query, retrieval_k=100, rerank_k=50):
        """
        Complete RAG pipeline: Retrieve, Rerank, and Generate.
        """
        # Step 1: Retrieve documents
        retrieved_docs = self.retrieve(query, top_k=retrieval_k)
        
        # Step 2: Rerank documents
        reranked_docs = self.rerank(query, retrieved_docs, top_k=rerank_k)
        
        # Step 3: Prepare context for generation
        contexts = "\n\n".join([doc['text'] for doc in reranked_docs])
        # Truncate context if it's too long
        contexts = contexts[:127900] if len(contexts) > 128000 else contexts
        
        # Step 4: Generate answer
        answer = self.generate(query, contexts)
        return answer
