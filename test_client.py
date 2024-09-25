# test_client.py

import socketio

# Create a Socket.IO client instance
sio = socketio.Client()

# Define event handlers
@sio.event
def connect():
    print('Connection established')

# ... rest of your client code ...

# Connect to the Socket.IO server
sio.connect('http://localhost:5000')
