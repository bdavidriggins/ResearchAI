from services.ollama_client import OllamaClient
from services.db_service import save_chat, save_llm_response

ollama = OllamaClient()

def handle_chat(session_id, query, context):
    response = ollama.generate_response(query, context)
    tokens_used = len(response.split())  # Simplified token count
    save_chat(session_id, query, response, context, tokens_used)
    save_llm_response(session_id, query, response, tokens_used, 0.5, "llama3.1", 0.7)
    return response
