from flask import Flask, request, jsonify
from db_service import DatabaseManager, configure_logging  # Adjusted the import to match the module
from rag_system import RAGSystem  # Import the RAGSystem class
import logging
import traceback

# Ensure logging is configured
configure_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the RAGSystem
try:
    rag_system = RAGSystem()
    logger.info("RAGSystem initialized successfully.", extra={'session_id': 'system'})
except Exception as e:
    error_message = str(e)
    stack_trace = traceback.format_exc()
    logger.error(f"Failed to initialize RAGSystem: {error_message}", extra={
        'session_id': 'system',
        'error_stack': stack_trace
    })
    raise  # Exit if RAGSystem cannot be initialized

# Ensure that the database is initialized
try:
    with DatabaseManager() as db:
        db.init_db()
except Exception as e:
    error_message = str(e)
    stack_trace = traceback.format_exc()
    logger.error(f"Failed to initialize database: {error_message}", extra={
        'session_id': 'system',
        'error_stack': stack_trace
    })
    raise  # Exit if database cannot be initialized

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json

        # Extract required parameters
        session_id = data.get('session_id')
        query = data.get('query')
        context = data.get('context', '')

        # Validate inputs
        if not session_id or not query:
            logger.error("Missing 'session_id' or 'query' in the request.", extra={'session_id': session_id or 'unknown'})
            return jsonify({"error": "Missing 'session_id' or 'query' in the request."}), 400

        try:
            # Generate response using RAGSystem
            response_text = rag_system.rag_pipeline(query)
        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logger.error(f"Error during RAG pipeline execution: {error_message}", extra={
                'session_id': session_id,
                'error_stack': stack_trace
            })
            return jsonify({"error": "An error occurred while processing your request."}), 500

        # Calculate tokens used and processing time (dummy values for illustration)
        tokens_used = len(response_text.split())
        processing_time = 0.1  # In seconds
        model_used = rag_system.ollama_model
        temperature = 0.7

        # Save the chat, log, and LLM response to the database
        try:
            with DatabaseManager() as db:
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
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logger.error(f"Database error: {error_message}", extra={
                'session_id': session_id,
                'error_stack': stack_trace
            })
            # Even if logging to the database fails, we can attempt to log the error
            try:
                with DatabaseManager() as db:
                    db.save_log(
                        session_id=session_id,
                        log_level="ERROR",
                        message="Database error during chat saving.",
                        error_stack=stack_trace
                    )
            except Exception as log_error:
                # If logging fails, output to console
                print(f"Failed to log to database: {log_error}")

            return jsonify({"error": "An error occurred while saving your request."}), 500

        # Return the generated response
        return jsonify({"response": response_text})

    except Exception as e:
        error_message = str(e)
        stack_trace = traceback.format_exc()
        session_id = request.json.get('session_id', 'unknown') if request.json else 'unknown'
        logger.error(f"Unexpected error in /chat endpoint: {error_message}", extra={
            'session_id': session_id,
            'error_stack': stack_trace
        })
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
