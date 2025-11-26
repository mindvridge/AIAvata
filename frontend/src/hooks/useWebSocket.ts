/**
 * WebSocket connection hook for direct avatar communication
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import type {
  ConnectionState,
  PipelineState,
  Emotion,
  WebSocketMessage
} from '../types';

interface UseWebSocketOptions {
  url: string;
  onMessage?: (message: WebSocketMessage) => void;
  onVideoFrame?: (data: ArrayBuffer) => void;
  autoReconnect?: boolean;
  reconnectInterval?: number;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  connectionState: ConnectionState;
  pipelineState: PipelineState;
  emotion: Emotion;
  connect: () => void;
  disconnect: () => void;
  sendAudio: (data: ArrayBuffer) => void;
  sendMessage: (message: WebSocketMessage) => void;
  error: Error | null;
}

export function useWebSocket(options: UseWebSocketOptions): UseWebSocketReturn {
  const {
    url,
    onMessage,
    onVideoFrame,
    autoReconnect = true,
    reconnectInterval = 3000,
  } = options;

  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [pipelineState, setPipelineState] = useState<PipelineState>('idle');
  const [emotion, setEmotion] = useState<Emotion>('neutral');
  const [error, setError] = useState<Error | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const isIntentionalDisconnect = useRef(false);

  const handleMessage = useCallback((event: MessageEvent) => {
    // Binary data is video frame
    if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
      if (event.data instanceof Blob) {
        event.data.arrayBuffer().then((buffer) => {
          onVideoFrame?.(buffer);
        });
      } else {
        onVideoFrame?.(event.data);
      }
      return;
    }

    // Text data is JSON message
    try {
      const message: WebSocketMessage = JSON.parse(event.data);

      // Handle status updates
      if (message.type === 'status') {
        const status = (message as { status: string }).status;
        if (status === 'processing' || status === 'speaking' ||
            status === 'listening' || status === 'idle') {
          setPipelineState(status as PipelineState);
        }
      }

      // Handle emotion changes
      if (message.type === 'emotion_changed') {
        const newEmotion = (message as { emotion: string }).emotion as Emotion;
        setEmotion(newEmotion);
      }

      // Handle errors
      if (message.type === 'error') {
        console.error('WebSocket error:', (message as { error: string }).error);
      }

      onMessage?.(message);
    } catch (err) {
      console.error('Failed to parse WebSocket message:', err);
    }
  }, [onMessage, onVideoFrame]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    isIntentionalDisconnect.current = false;
    setError(null);
    setConnectionState('connecting');

    try {
      const ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        console.log('WebSocket connected');
        setConnectionState('connected');
        setError(null);
      };

      ws.onmessage = handleMessage;

      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError(new Error('WebSocket connection error'));
        setConnectionState('error');
      };

      ws.onclose = () => {
        console.log('WebSocket closed');
        setConnectionState('disconnected');
        wsRef.current = null;

        // Auto reconnect
        if (autoReconnect && !isIntentionalDisconnect.current) {
          console.log(`Reconnecting in ${reconnectInterval}ms...`);
          reconnectTimeoutRef.current = window.setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };

      wsRef.current = ws;
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      setError(err instanceof Error ? err : new Error('Failed to connect'));
      setConnectionState('error');
    }
  }, [url, handleMessage, autoReconnect, reconnectInterval]);

  const disconnect = useCallback(() => {
    isIntentionalDisconnect.current = true;

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnectionState('disconnected');
  }, []);

  const sendAudio = useCallback((data: ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    }
  }, []);

  const sendMessage = useCallback((message: WebSocketMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isIntentionalDisconnect.current = true;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return {
    isConnected: connectionState === 'connected',
    connectionState,
    pipelineState,
    emotion,
    connect,
    disconnect,
    sendAudio,
    sendMessage,
    error,
  };
}

export default useWebSocket;
