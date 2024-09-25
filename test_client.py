# test_client.py

import socketio
import uuid
import time

# Initialize SocketIO client
sio = socketio.Client()

# Generate a unique session ID for this client
SESSION_ID = str(uuid.uuid4())

# Event handler for connection
@sio.event
def connect():
    print("Successfully connected to the server.")

    # Prepare the message data
    message_data = {
        "session_id": SESSION_ID,
        "query": "Who was Robert Rogers?"
    }

    print(f"Sending message: {message_data['query']}")
    sio.emit('chat message', message_data)

# Event handler for disconnection
@sio.event
def disconnect():
    print("Disconnected from the server.")

# Event handler for receiving response chunks
@sio.on('response')
def on_response(data):
    chunk = data.get('chunk', '')
    print(f"Received chunk: {chunk}")

# Event handler for response completion
@sio.on('response_complete')
def on_response_complete():
    print("Response streaming completed.")
    sio.disconnect()

# Event handler for errors
@sio.on('error')
def on_error(data):
    error_message = data.get('error', 'Unknown error')
    print(f"Error from server: {error_message}")

def main():
    try:
        # Connect to the server
        sio.connect('http://localhost:5000')

        # Wait for the communication to complete
        sio.wait()

    except socketio.exceptions.ConnectionError as e:
        print(f"Connection failed: {e}")

if __name__ == '__main__':
    main()
