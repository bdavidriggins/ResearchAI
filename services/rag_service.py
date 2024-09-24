import os
import json
import logging
import requests
import numpy as np
import torch
import faiss
import traceback
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self,
                 embedding_model_name='multi-qa-mpnet-base-dot-v1',
                 reranker_model_name='cross-encoder/ms-marco-TinyBERT-L-6',
                 ollama_api_url='http://localhost:11434/api/generate',
                 ollama_model="llama2-uncensored"):
        """
        Initialize the Retrieval-Augmented Generation (RAG) system.
        """
        try:
            # Paths
            # Get the absolute path to the directory where the script is located
            file_path = os.path.dirname(os.path.realpath(__file__))
            app_dir = os.path.dirname(file_path)

            # Construct absolute paths based on the app's directory
            embeddings_dir = os.path.join(app_dir, 'faiss')
            faiss_index_path = os.path.join(embeddings_dir, 'faiss_index.bin')
            faiss_metadata_path = os.path.join(embeddings_dir, 'faiss_metadata.json')

            # Load FAISS index
            self.index = faiss.read_index(faiss_index_path)
            logger.info("FAISS index loaded from '%s'.", faiss_index_path)

            # Load metadata
            with open(faiss_metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info("FAISS metadata loaded from '%s'.", faiss_metadata_path)

            # Initialize embedding model
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logger.info("Embedding model '%s' loaded.", embedding_model_name)

            # Initialize reranker model
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
            self.reranker_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.reranker_model.to(self.reranker_device)
            logger.info("Reranker model '%s' loaded on device '%s'.", reranker_model_name, self.reranker_device)

            # Initialize Ollama client parameters
            self.ollama_api_url = ollama_api_url
            self.ollama_model = ollama_model
            logger.info("Ollama client initialized with API URL '%s' and model '%s'.", ollama_api_url, ollama_model)

        except Exception as e:
            logger.error("Error during RAGSystem initialization: %s", str(e))
            raise e  # Re-raise exception after logging

    def retrieve(self, query, top_k=100):
        """
        Retrieve top_k documents from the FAISS index based on the query.
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).astype('float32')
            # Search FAISS index
            distances, indices = self.index.search(np.array([query_embedding]), top_k)
            # Retrieve corresponding documents
            retrieved_docs = [self.metadata[idx] for idx in indices[0]]
            logger.info("Retrieved %d documents for query '%s'.", len(retrieved_docs), query)
            return retrieved_docs
        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()  # Capture the full stack trace as a string
            logger.error("Error during retrieval: %s\nStack trace: %s", error_message, stack_trace)
            return []

    def rerank(self, query, candidates, top_k=5):
        """
        Rerank the retrieved documents using a cross-encoder model.
        """
        try:
            # Prepare input pairs for reranker
            inputs = [f"{query} [SEP] {candidate['text']}" for candidate in candidates]
            # Tokenize inputs
            encoded = self.reranker_tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
            encoded = {k: v.to(self.reranker_device) for k, v in encoded.items()}

            # Perform inference
            with torch.no_grad():
                outputs = self.reranker_model(**encoded)
                scores = outputs.logits.squeeze().cpu().numpy()

            # Assign scores to candidates
            for i, candidate in enumerate(candidates):
                candidate['score'] = scores[i]

            # Sort candidates by score
            sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
            logger.info("Reranked documents for query '%s'.", query)
            return sorted_candidates[:top_k]
        except Exception as e:
            logger.error("Error during reranking: %s", str(e))
            return candidates[:top_k]  # Return top_k candidates even if reranking fails

    def generate(self, prompt, context):
        """
        Generate a response using the Ollama API.
        """
        try:
            # Construct the full prompt
            full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
            payload = {
                "model": self.ollama_model,
                "prompt": full_prompt,
                "max_tokens": 4096,
                "temperature": 0.7,
                "stream": False
            }
            # Send request to Ollama API
            response = requests.post(self.ollama_api_url, json=payload)
            response.raise_for_status()
            # Extract the response
            answer = response.json().get('response', '')
            logger.info("Generated response for prompt '%s'.", prompt)
            return answer
        except requests.exceptions.RequestException as e:
            logger.error("Error communicating with Ollama API: %s", str(e))
            return "An error occurred while generating the response."
        except Exception as e:
            logger.error("Unexpected error during generation: %s", str(e))
            return "An error occurred while generating the response."

    def rag_pipeline(self, query, retrieval_k=100, rerank_k=50):
        """
        Complete RAG pipeline: Retrieve, Rerank, and Generate.
        """
        try:
            # Step 1: Retrieve documents
            retrieved_docs = self.retrieve(query, top_k=retrieval_k)
            if not retrieved_docs:
                logger.warning("No documents retrieved for query '%s'.", query)
                return "No relevant documents found."

            # Step 2: Rerank documents
            reranked_docs = self.rerank(query, retrieved_docs, top_k=rerank_k)

            # Step 3: Prepare context for generation
            contexts = "\n\n".join([doc['text'] for doc in reranked_docs])
            # Truncate context if it's too long
            max_context_length = 127900
            if len(contexts) > max_context_length:
                contexts = contexts[:max_context_length]
                logger.warning("Context truncated to %d characters.", max_context_length)

            # Step 4: Generate answer
            answer = self.generate(query, contexts)
            return answer
        except Exception as e:
            logger.error("Error in RAG pipeline: %s", str(e))
            return "An error occurred while processing your request."
