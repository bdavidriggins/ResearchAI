# test_client.py

import socketio
import logging
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a Socket.IO client instance
sio = socketio.Client(logger=True)

# Define event handlers

@sio.event
def connect():
    logger.info('Connection established')
    test_send_message()

@sio.event
def disconnect():
    logger.info('Disconnected from server')

@sio.on('response')
def on_response(data):
    logger.info(f"Received response chunk: {data.get('chunk')}")
    
@sio.on('response_complete')
def on_response_complete():
    logger.info('Response streaming complete')

@sio.on('error')
def on_error(error):
    logger.error(f"Error received from server: {error.get('error')}")

def test_send_message():
    """
    Test the 'chat message' functionality by sending a test message.
    """
    try:
        logger.info("Sending 'chat message' event to server...")
        data = {
            'session_id': 'test_session',
            'query': 'Who was Robert Rogers?',
            'context': 'Your are a research assistant'
        }
        sio.emit('chat message', data)
        logger.debug(f"Message sent with data: {data}")
    except Exception as e:
        logger.error(f"Error while sending message: {str(e)}")

def test_invalid_message():
    """
    Test sending an invalid message (missing required 'query' field) to simulate error handling.
    """
    try:
        logger.info("Sending invalid 'chat message' event to server (missing query)...")
        data = {
            'session_id': 'test_session_invalid',
            'context': 'Geography'
        }
        sio.emit('chat message', data)
        logger.debug(f"Invalid message sent with data: {data}")
    except Exception as e:
        logger.error(f"Error while sending invalid message: {str(e)}")

def test_reconnection():
    """
    Test automatic reconnection to the server.
    """
    try:
        logger.info("Testing reconnection by disconnecting and reconnecting...")
        sio.disconnect()
        time.sleep(2)
        sio.connect('http://localhost:5000')
        logger.info("Reconnected successfully.")
    except Exception as e:
        logger.error(f"Error during reconnection: {str(e)}")

# Connect to the Socket.IO server
try:
    logger.info("Connecting to the Socket.IO server at 'http://localhost:5000'...")
    sio.connect('http://localhost:5000')

    # Wait for events to be handled
    time.sleep(10)  # Adjust the sleep time as needed to allow interactions to complete

    # Test an invalid message
    #test_invalid_message()

    # Test reconnection
    #test_reconnection()

except Exception as e:
    logger.error(f"Error during connection or communication: {str(e)}")

finally:
    logger.info("Disconnecting from server...")
    sio.disconnect()
