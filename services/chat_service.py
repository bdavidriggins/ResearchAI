from services.ollama_client import OllamaClient
from services.db_service import DatabaseManager  # Import the DatabaseManager class

ollama = OllamaClient()

def handle_chat(session_id, query, context):
    # Generate response from the Ollama model
    response = ollama.generate_response(query, context)
    tokens_used = len(response.split())  # Simplified token count

    # Save the chat and LLM response using the DatabaseManager
    with DatabaseManager() as db:
        db.save_chat(session_id, query, response, context, tokens_used)
        db.save_llm_response(session_id, query, response, tokens_used, 0.5, "llama3.1", 0.7)
    
    return response
