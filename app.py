# app.py

import eventlet
eventlet.monkey_patch()

# Now import other modules
from flask import Flask, request
from flask_socketio import SocketIO, emit
from services.db_service import DatabaseManager, configure_logging
from services.rag_service import RAGSystem  # Import the RAGSystem class
import logging
import traceback

# Ensure logging is configured
configure_logging()
logger = logging.getLogger(__name__)

# Initialize Flask app and SocketIO
logger.debug("Initializing Flask app and SocketIO...")
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', ping_timeout=120, ping_interval=25)
logger.debug("Flask app and SocketIO initialized.")

# Initialize the RAGSystem
try:
    logger.info("Starting initialization of RAGSystem...", extra={'session_id': 'system'})
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
    logger.info("Starting database initialization...", extra={'session_id': 'system'})
    with DatabaseManager() as db:
        db.init_db()
    logger.info("Database initialized successfully.", extra={'session_id': 'system'})
except Exception as e:
    error_message = str(e)
    stack_trace = traceback.format_exc()
    logger.error(f"Failed to initialize database: {error_message}", extra={
        'session_id': 'system',
        'error_stack': stack_trace
    })
    raise  # Exit if database cannot be initialized

@socketio.on('connect')
def on_connect(auth):
    logger.info("Client connected.", extra={'session_id': request.sid})

@socketio.on('disconnect')
def on_disconnect():
    logger.info("Client disconnected.", extra={'session_id': request.sid})

@socketio.on('chat message')
def handle_message(data):
    logger.info(f"Received 'chat message' event.", extra={'session_id': data.get('session_id', 'unknown')})
    try:
        # Extract required parameters
        session_id = data.get('session_id')
        query = data.get('query')
        context = data.get('context', '')

        logger.debug(f"Extracted data - session_id: {session_id}, query: {query}, context: {context}")

        # Validate inputs
        if not session_id or not query:
            logger.error("Missing 'session_id' or 'query' in the message.",
                         extra={'session_id': session_id or 'unknown'})
            emit('error', {"error": "Missing 'session_id' or 'query' in the message."})
            return

        # Get a generator for streaming response chunks
        logger.info(f"Starting RAG pipeline stream for query: '{query}'", extra={'session_id': session_id})
        response_generator = rag_system.rag_pipeline_stream(query)
        logger.debug("Received response generator from RAGSystem.", extra={'session_id': session_id})

        # Initialize variables to collect response and stats
        response_text = ""
        tokens_used = 0
        processing_time = 0  # Placeholder for actual processing time
        model_used = rag_system.ollama_model
        temperature = 0.7

        # Stream each chunk to the client
        logger.info("Streaming response chunks to client...", extra={'session_id': session_id})
        for chunk in response_generator:
            logger.debug(f"Received chunk: '{chunk}'", extra={'session_id': session_id})
            response_text += chunk
            tokens_used += len(chunk.split())  # Simplistic token count
            emit('response', {'chunk': chunk})
            logger.debug("Emitted chunk to client.", extra={'session_id': session_id})

        # After streaming all chunks, send a completion message
        logger.info("Finished streaming response chunks.", extra={'session_id': session_id})
        emit('response_complete')
        logger.info("Emitted 'response_complete' to client.", extra={'session_id': session_id})

        # Save the chat, log, and LLM response to the database
        logger.info("Saving chat and response data to database...", extra={'session_id': session_id})
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
                logger.debug("Chat record saved.", extra={'session_id': session_id})

                # Save a log record
                db.save_log(
                    session_id=session_id,
                    log_level="INFO",
                    message="Chat handled successfully."
                )
                logger.debug("Log record saved.", extra={'session_id': session_id})

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
                logger.debug("LLM response record saved.", extra={'session_id': session_id})

            logger.info("Data saved to database successfully.", extra={'session_id': session_id})
        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logger.error(f"Database error: {error_message}",
                         extra={'session_id': session_id, 'error_stack': stack_trace})

            # Attempt to log the error in the database
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

            emit('error', {"error": "An error occurred while saving your request."})
            return

    except Exception as e:
        error_message = str(e)
        stack_trace = traceback.format_exc()
        session_id = data.get('session_id', 'unknown') if data else 'unknown'
        logger.error(f"Unexpected error in handle_message: {error_message}",
                     extra={'session_id': session_id, 'error_stack': stack_trace})
        emit('error', {"error": "An internal server error occurred."})



if __name__ == '__main__':
    logger.info("Starting Socket.IO server...")
    socketio.run(app, host='0.0.0.0', port=5000)
    logger.info("Socket.IO server started...")
