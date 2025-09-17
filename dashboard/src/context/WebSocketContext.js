import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { useAuth } from './AuthContext';

const WebSocketContext = createContext();

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

export const WebSocketProvider = ({ children }) => {
  const [ws, setWs] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [notifications, setNotifications] = useState([]);
  const [messageQueue, setMessageQueue] = useState([]);
  const { isAuthenticated, token } = useAuth();

  useEffect(() => {
    if (isAuthenticated && token) {
      connect();
    } else {
      disconnect();
    }

    return () => {
      disconnect();
    };
  }, [isAuthenticated, token]);

  const connect = () => {
    const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws';
    const websocket = new WebSocket(wsUrl);

    websocket.onopen = () => {
      console.log('WebSocket connected');
      setConnectionStatus('connected');
      setWs(websocket);

      // Send authentication
      websocket.send(JSON.stringify({
        type: 'auth',
        token: token,
      }));

      // Send queued messages
      messageQueue.forEach(msg => {
        websocket.send(JSON.stringify(msg));
      });
      setMessageQueue([]);
    };

    websocket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        handleMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('error');
    };

    websocket.onclose = () => {
      console.log('WebSocket disconnected');
      setConnectionStatus('disconnected');
      setWs(null);

      // Attempt to reconnect after 5 seconds
      if (isAuthenticated) {
        setTimeout(() => {
          connect();
        }, 5000);
      }
    };
  };

  const disconnect = () => {
    if (ws) {
      ws.close();
      setWs(null);
    }
  };

  const handleMessage = (message) => {
    switch (message.type) {
      case 'notification':
        addNotification(message.data);
        break;
      
      case 'command_completed':
        addNotification({
          title: 'Command Completed',
          message: `Command "${message.data.command}" completed in ${message.data.execution_time}s`,
          type: 'success',
          timestamp: new Date().toISOString(),
        });
        break;
      
      case 'incident_resolved':
        addNotification({
          title: 'Incident Resolved',
          message: `Incident has been resolved: ${message.data.incident.description}`,
          type: 'info',
          timestamp: new Date().toISOString(),
        });
        break;
      
      case 'alert':
        addNotification({
          title: 'Alert',
          message: message.data.message,
          type: message.data.severity || 'warning',
          timestamp: new Date().toISOString(),
        });
        break;
      
      case 'pong':
        // Keep-alive response
        break;
      
      default:
        console.log('Unknown message type:', message.type);
    }
  };

  const addNotification = (notification) => {
    const newNotification = {
      ...notification,
      id: Date.now(),
      read: false,
    };
    setNotifications(prev => [newNotification, ...prev].slice(0, 50)); // Keep last 50
  };

  const sendMessage = useCallback((message) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    } else {
      // Queue message if not connected
      setMessageQueue(prev => [...prev, message]);
    }
  }, [ws]);

  const markNotificationAsRead = (id) => {
    setNotifications(prev => 
      prev.map(n => n.id === id ? { ...n, read: true } : n)
    );
  };

  const clearNotifications = () => {
    setNotifications([]);
  };

  const value = {
    connectionStatus,
    notifications,
    sendMessage,
    markNotificationAsRead,
    clearNotifications,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};