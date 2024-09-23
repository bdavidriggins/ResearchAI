from flask import Flask, request, jsonify
from services.db_service import DatabaseManager  # Adjusted the import to match the module
from services.chat_service import handle_chat  # Add this import

app = Flask(__name__)

# Ensure that the database is initialized
with DatabaseManager() as db:
    db.init_db()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    session_id = data.get('session_id')
    query = data.get('query')
    context = data.get('context', '')

    # Call the chat handling function to get the real response
    response_text = handle_chat(session_id, query, context)

    # Simulate tokens used and processing time
    tokens_used = len(query.split())
    processing_time = 0.1  # In seconds
    model_used = 'llama2-uncensored'
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
            db.save_log(
                session_id=session_id,
                log_level="ERROR",
                message=str(e),
                error_stack=str(e)
            )
            return jsonify({"error": "An error occurred while processing your request."}), 500

    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
