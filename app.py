from flask import Flask, request, jsonify
from services.db_service import DatabaseManager  # Adjusted the import to match the module
from services.rag_service import RAGSystem  # Import the RAGSystem class
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the RAGSystem
try:
    rag_system = RAGSystem()
    logger.info("RAGSystem initialized successfully.")
except Exception as e:
    logger.error("Failed to initialize RAGSystem: %s", str(e))
    raise e  # Exit if RAGSystem cannot be initialized

# Ensure that the database is initialized
with DatabaseManager() as db:
    db.init_db()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json

    # Extract required parameters
    session_id = data.get('session_id')
    query = data.get('query')
    context = data.get('context', '')

    # Validate inputs
    if not session_id or not query:
        logger.error("Missing 'session_id' or 'query' in the request.")
        return jsonify({"error": "Missing 'session_id' or 'query' in the request."}), 400

    try:
        # Generate response using RAGSystem
        response_text = rag_system.rag_pipeline(query)
    except Exception as e:
        logger.error("Error during RAG pipeline execution: %s", str(e))
        return jsonify({"error": "An error occurred while processing your request."}), 500

    # Calculate tokens used and processing time (dummy values for illustration)
    tokens_used = len(response_text.split())
    processing_time = 0.1  # In seconds
    model_used = rag_system.ollama_model
    temperature = 0.7

    # Save the chat, log, and LLM response to the database
    with DatabaseManager() as db:
        try:
            # Save the chat record
            db.save_chat(
                session_id=session_id,
                query=query,
                response=response_text,
                context=context,
                tokens_used=tokens_used
            )

            # Save a log record
            db.save_log(
                session_id=session_id,
                log_level="INFO",
                message="Chat handled successfully."
            )

            # Save the LLM response record
            db.save_llm_response(
                session_id=session_id,
                prompt=query,
                response_text=response_text,
                tokens_used=tokens_used,
                processing_time=processing_time,
                model_used=model_used,
                temperature=temperature
            )
        except Exception as e:
            # Log the error
            logger.error("Database error: %s", str(e))
            db.save_log(
                session_id=session_id,
                log_level="ERROR",
                message="Database error.",
                error_stack=str(e)
            )
            return jsonify({"error": "An error occurred while saving your request."}), 500

    # Return the generated response
    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
