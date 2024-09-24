// services/socketService.js

// Import the Socket.IO client library
import { io } from 'socket.io-client';

// Establish a connection to the WebSocket server
export const socket = io('http://localhost:5000', {
  transports: ['websocket'], // Use WebSocket transport
});

// Handle connection errors
socket.on('connect_error', (err) => {
  console.error('Connection error:', err);
});
