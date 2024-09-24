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

# Import the DatabaseManager and configure_logging from db_service
from db_service import DatabaseManager, configure_logging

# Ensure logging is configured
configure_logging()
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self,
                 embedding_model_name='multi-qa-mpnet-base-dot-v1',
                 reranker_model_name='cross-encoder/ms-marco-TinyBERT-L-6',
                 ollama_api_url='http://localhost:11434/api/generate',
                 ollama_model="llama2-uncensored",
                 session_id='default_session'):
        """
        Initialize the Retrieval-Augmented Generation (RAG) system.
        """
        self.session_id = session_id  # Added session_id for logging

        try:
            # Paths
            file_path = os.path.dirname(os.path.realpath(__file__))
            app_dir = os.path.dirname(file_path)

            # Construct absolute paths based on the app's directory
            embeddings_dir = os.path.join(app_dir, 'faiss')
            faiss_index_path = os.path.join(embeddings_dir, 'faiss_index.bin')
            faiss_metadata_path = os.path.join(embeddings_dir, 'faiss_metadata.json')

            # Load FAISS index
            self.index = faiss.read_index(faiss_index_path)
            logger.info(f"FAISS index loaded from '{faiss_index_path}'.", extra={'session_id': self.session_id})

            # Load metadata
            with open(faiss_metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"FAISS metadata loaded from '{faiss_metadata_path}'.", extra={'session_id': self.session_id})

            # Initialize embedding model
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logger.info(f"Embedding model '{embedding_model_name}' loaded.", extra={'session_id': self.session_id})

            # Initialize reranker model
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
            self.reranker_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.reranker_model.to(self.reranker_device)
            logger.info(f"Reranker model '{reranker_model_name}' loaded on device '{self.reranker_device}'.", extra={'session_id': self.session_id})

            # Initialize Ollama client parameters
            self.ollama_api_url = ollama_api_url
            self.ollama_model = ollama_model
            logger.info(f"Ollama client initialized with API URL '{ollama_api_url}' and model '{ollama_model}'.", extra={'session_id': self.session_id})

        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logger.error(f"Error during RAGSystem initialization: {error_message}", extra={
                'session_id': self.session_id,
                'error_stack': stack_trace
            })
            raise

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
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query '{query}'.", extra={'session_id': self.session_id})
            return retrieved_docs
        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logger.error(f"Error during retrieval: {error_message}", extra={
                'session_id': self.session_id,
                'error_stack': stack_trace
            })
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
            logger.info(f"Reranked documents for query '{query}'.", extra={'session_id': self.session_id})
            return sorted_candidates[:top_k]
        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logger.error(f"Error during reranking: {error_message}", extra={
                'session_id': self.session_id,
                'error_stack': stack_trace
            })
            return candidates[:top_k]

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
            logger.info(f"Generated response for prompt '{prompt}'.", extra={'session_id': self.session_id})
            return answer
        except requests.exceptions.RequestException as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logger.error(f"Error communicating with Ollama API: {error_message}", extra={
                'session_id': self.session_id,
                'error_stack': stack_trace,
                'request_data': str(payload)
            })
            return "An error occurred while generating the response."
        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logger.error(f"Unexpected error during generation: {error_message}", extra={
                'session_id': self.session_id,
                'error_stack': stack_trace
            })
            return "An error occurred while generating the response."

    def rag_pipeline(self, query, retrieval_k=100, rerank_k=50):
        """
        Complete RAG pipeline: Retrieve, Rerank, and Generate.
        """
        try:
            # Step 1: Retrieve documents
            retrieved_docs = self.retrieve(query, top_k=retrieval_k)
            if not retrieved_docs:
                logger.warning(f"No documents retrieved for query '{query}'.", extra={'session_id': self.session_id})
                return "No relevant documents found."

            # Step 2: Rerank documents
            reranked_docs = self.rerank(query, retrieved_docs, top_k=rerank_k)

            # Step 3: Prepare context for generation
            contexts = "\n\n".join([doc['text'] for doc in reranked_docs])
            # Truncate context if it's too long
            max_context_length = 127900
            if len(contexts) > max_context_length:
                contexts = contexts[:max_context_length]
                logger.warning(f"Context truncated to {max_context_length} characters.", extra={'session_id': self.session_id})

            # Step 4: Generate answer
            answer = self.generate(query, contexts)
            return answer
        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logger.error(f"Error in RAG pipeline: {error_message}", extra={
                'session_id': self.session_id,
                'error_stack': stack_trace
            })
            return "An error occurred while processing your request."
