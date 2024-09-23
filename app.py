from flask import Flask, request, jsonify
from services.chat_service import handle_chat

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    session_id = data.get('session_id')
    query = data.get('query')
    context = data.get('context', '')

    response = handle_chat(session_id, query, context)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)