# /services/rag_system.py

import os
import json
import logging
import requests
import numpy as np
import torch
import faiss
import traceback
from typing import List, Dict, Any, Generator, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .db_service import DatabaseManager, configure_logging

# Constants
MAX_CONTEXT_LENGTH = 127900
DEFAULT_EMBEDDING_MODEL = 'multi-qa-mpnet-base-dot-v1'
DEFAULT_RERANKER_MODEL = 'cross-encoder/ms-marco-TinyBERT-L-6'
DEFAULT_OLLAMA_API_URL = 'http://localhost:11434/api/generate'
DEFAULT_OLLAMA_MODEL = "llama2-uncensored"
DEFAULT_SESSION_ID = 'default_session'

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model_name: str = DEFAULT_RERANKER_MODEL,
        ollama_api_url: str = DEFAULT_OLLAMA_API_URL,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        session_id: str = DEFAULT_SESSION_ID
    ) -> None:
        """
        Initialize the Retrieval-Augmented Generation (RAG) system.

        Args:
            embedding_model_name (str): Name of the embedding model.
            reranker_model_name (str): Name of the reranker model.
            ollama_api_url (str): URL of the Ollama API.
            ollama_model (str): Name of the Ollama model.
            session_id (str): Identifier for the session.
        """
        self.session_id = session_id

        try:
            # Determine paths
            file_path = os.path.dirname(os.path.realpath(__file__))
            app_dir = os.path.dirname(file_path)
            embeddings_dir = os.path.join(app_dir, 'faiss')
            faiss_index_path = os.path.join(embeddings_dir, 'faiss_index.bin')
            faiss_metadata_path = os.path.join(embeddings_dir, 'faiss_metadata.json')

            # Load FAISS index
            self.index: faiss.Index = faiss.read_index(faiss_index_path)
            logger.info(
                f"FAISS index loaded from '{faiss_index_path}'.",
                extra={'session_id': self.session_id}
            )

            # Load metadata
            with open(faiss_metadata_path, 'r') as f:
                self.metadata: List[Dict[str, Any]] = json.load(f)
            logger.info(
                f"FAISS metadata loaded from '{faiss_metadata_path}'.",
                extra={'session_id': self.session_id}
            )

            # Initialize embedding model
            self.embedding_model: SentenceTransformer = SentenceTransformer(embedding_model_name)
            logger.info(
                f"Embedding model '{embedding_model_name}' loaded.",
                extra={'session_id': self.session_id}
            )

            # Initialize reranker model
            self.reranker_tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
            self.reranker_model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
            self.reranker_device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.reranker_model.to(self.reranker_device)
            logger.info(
                f"Reranker model '{reranker_model_name}' loaded on device '{self.reranker_device}'.",
                extra={'session_id': self.session_id}
            )

            # Initialize Ollama client parameters
            self.ollama_api_url: str = ollama_api_url
            self.ollama_model: str = ollama_model
            logger.info(
                f"Ollama client initialized with API URL '{ollama_api_url}' and model '{ollama_model}'.",
                extra={'session_id': self.session_id}
            )

        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logger.error(
                f"Error during RAGSystem initialization: {error_message}",
                extra={
                    'session_id': self.session_id,
                    'error_stack': stack_trace
                }
            )
            raise

    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve top_k documents from the FAISS index based on the query.

        Args:
            query (str): The input query string.
            top_k (int): Number of top documents to retrieve.

        Returns:
            List[Dict[str, Any]]: Retrieved documents.
        """
        try:
            query_embedding = self.embedding_model.encode(query).astype('float32')
            distances, indices = self.index.search(np.array([query_embedding]), top_k)
            retrieved_docs = [self.metadata[idx] for idx in indices[0] if idx < len(self.metadata)]
            logger.info(
                f"Retrieved {len(retrieved_docs)} documents for query '{query}'.",
                extra={'session_id': self.session_id}
            )
            return retrieved_docs
        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logger.error(
                f"Error during retrieval: {error_message}",
                extra={
                    'session_id': self.session_id,
                    'error_stack': stack_trace
                }
            )
            return []

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank the retrieved documents using a cross-encoder model.

        Args:
            query (str): The input query string.
            candidates (List[Dict[str, Any]]): Retrieved documents to rerank.
            top_k (int): Number of top documents to return after reranking.

        Returns:
            List[Dict[str, Any]]: Reranked documents.
        """
        try:
            inputs = [f"{query} [SEP] {candidate['text']}" for candidate in candidates]
            encoded = self.reranker_tokenizer(
                inputs, padding=True, truncation=True, return_tensors='pt'
            )
            encoded = {k: v.to(self.reranker_device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.reranker_model(**encoded)
                scores = outputs.logits.squeeze().cpu().numpy()

            for i, candidate in enumerate(candidates):
                candidate['score'] = scores[i]

            sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
            logger.info(
                f"Reranked documents for query '{query}'.",
                extra={'session_id': self.session_id}
            )
            return sorted_candidates[:top_k]
        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logger.error(
                f"Error during reranking: {error_message}",
                extra={
                    'session_id': self.session_id,
                    'error_stack': stack_trace
                }
            )
            return candidates[:top_k]

    def generate(self, prompt: str, context: str) -> str:
        """
        Generate a response using the Ollama API.

        Args:
            prompt (str): The input prompt or question.
            context (str): The context retrieved from documents.

        Returns:
            str: Generated answer.
        """
        try:
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
            answer = response.json().get('response', '')
            logger.info(
                f"Generated response for prompt '{prompt}'.",
                extra={'session_id': self.session_id}
            )
            return answer
        except requests.exceptions.RequestException as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logger.error(
                f"Error communicating with Ollama API: {error_message}",
                extra={
                    'session_id': self.session_id,
                    'error_stack': stack_trace,
                    'request_data': json.dumps(payload)
                }
            )
            return "An error occurred while generating the response."
        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logger.error(
                f"Unexpected error during generation: {error_message}",
                extra={
                    'session_id': self.session_id,
                    'error_stack': stack_trace
                }
            )
            return "An error occurred while generating the response."

    def rag_pipeline(self, query: str, retrieval_k: int = 100, rerank_k: int = 50) -> str:
        """
        Complete RAG pipeline: Retrieve, Rerank, and Generate.

        Args:
            query (str): The input query string.
            retrieval_k (int): Number of documents to retrieve.
            rerank_k (int): Number of documents to rerank.

        Returns:
            str: Generated answer.
        """
        try:
            retrieved_docs = self.retrieve(query, top_k=retrieval_k)
            if not retrieved_docs:
                logger.warning(
                    f"No documents retrieved for query '{query}'.",
                    extra={'session_id': self.session_id}
                )
                return "No relevant documents found."

            reranked_docs = self.rerank(query, retrieved_docs, top_k=rerank_k)

            contexts = "\n\n".join([doc['text'] for doc in reranked_docs])
            if len(contexts) > MAX_CONTEXT_LENGTH:
                contexts = contexts[:MAX_CONTEXT_LENGTH]
                logger.warning(
                    f"Context truncated to {MAX_CONTEXT_LENGTH} characters.",
                    extra={'session_id': self.session_id}
                )

            answer = self.generate(query, contexts)
            return answer
        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logger.error(
                f"Error in RAG pipeline: {error_message}",
                extra={
                    'session_id': self.session_id,
                    'error_stack': stack_trace
                }
            )
            return "An error occurred while processing your request."

    def rag_pipeline_stream(self, query: str, retrieval_k: int = 100, rerank_k: int = 50) -> Generator[str, None, None]:
        """
        Complete RAG pipeline with streaming: Retrieve, Rerank, and Generate.
        Yields intermediate status messages and streamed responses.

        Args:
            query (str): The input query string.
            retrieval_k (int): Number of documents to retrieve.
            rerank_k (int): Number of documents to rerank.

        Yields:
            Generator[str, None, None]: Streamed response chunks.
        """
        try:
            yield "Starting retrieval..."
            retrieved_docs = self.retrieve(query, top_k=retrieval_k)
            if not retrieved_docs:
                logger.warning(
                    f"No documents retrieved for query '{query}'.",
                    extra={'session_id': self.session_id}
                )
                yield "No relevant documents found."
                return

            yield "Retrieval complete. Starting reranking..."

            reranked_docs = self.rerank(query, retrieved_docs, top_k=rerank_k)

            yield "Reranking complete. Generating answer..."

            contexts = "\n\n".join([doc['text'] for doc in reranked_docs])
            if len(contexts) > MAX_CONTEXT_LENGTH:
                contexts = contexts[:MAX_CONTEXT_LENGTH]
                logger.warning(
                    f"Context truncated to {MAX_CONTEXT_LENGTH} characters.",
                    extra={'session_id': self.session_id}
                )
                yield f"Context truncated to {MAX_CONTEXT_LENGTH} characters."

            for chunk in self.generate_stream(query, contexts):
                yield chunk

        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logger.error(
                f"Error in RAG pipeline (stream): {error_message}",
                extra={
                    'session_id': self.session_id,
                    'error_stack': stack_trace
                }
            )
            yield "An error occurred while processing your request."

    def generate_stream(self, prompt: str, context: str) -> Generator[str, None, None]:
        try:
            full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
            payload = {
                "model": self.ollama_model,
                "prompt": full_prompt,
                "max_tokens": 4096,
                "temperature": 0.7,
                "stream": True
            }
            logger.info("Sending request to Ollama API for streaming generation.", extra={'session_id': self.session_id})
            with requests.post(self.ollama_api_url, json=payload, stream=True, timeout=120) as response:
                response.raise_for_status()
                logger.info("Received response from Ollama API.", extra={'session_id': self.session_id})
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        logger.debug(f"Received chunk from Ollama API: {decoded_line}", extra={'session_id': self.session_id})
                        yield decoded_line
            logger.info("Completed streaming from Ollama API.", extra={'session_id': self.session_id})
        except requests.exceptions.Timeout:
            logger.error("Ollama API request timed out.", extra={'session_id': self.session_id})
            yield "The request timed out while generating the response."
        except requests.exceptions.RequestException as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logger.error(
                f"Error communicating with Ollama API: {error_message}",
                extra={
                    'session_id': self.session_id,
                    'error_stack': stack_trace,
                    'request_data': json.dumps(payload)
                }
            )
            yield "An error occurred while generating the response."
        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logger.error(
                f"Unexpected error during streaming generation: {error_message}",
                extra={
                    'session_id': self.session_id,
                    'error_stack': stack_trace
                }
            )
            yield "An error occurred while generating the response."