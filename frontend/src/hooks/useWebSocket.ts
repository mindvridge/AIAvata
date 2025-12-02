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
  connect: (customUrl?: string) => void;
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
  const urlRef = useRef<string>(url); // URL을 ref로 관리하여 항상 최신 값 사용

  // URL prop이 변경되면 ref 업데이트
  useEffect(() => {
    urlRef.current = url;
  }, [url]);

  const handleMessage = useCallback((event: MessageEvent) => {
    // Binary data is video frame
    if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
      console.debug('Received video frame, size:', event.data instanceof ArrayBuffer ? event.data.byteLength : 'Blob');
      if (event.data instanceof Blob) {
        event.data.arrayBuffer().then((buffer) => {
          console.debug('Converted Blob to ArrayBuffer, size:', buffer.byteLength);
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

      // Handle ping/pong heartbeat
      if (message.type === 'ping') {
        // 서버 ping에 대해 pong 응답
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: 'pong' }));
        }
        return;
      }

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

  const connect = useCallback((customUrl?: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    // 커스텀 URL이 제공되면 사용하고, 없으면 ref의 최신 URL 사용
    const targetUrl = customUrl || urlRef.current;
    
    if (!targetUrl) {
      console.error('WebSocket URL is not set');
      setError(new Error('WebSocket URL is not set'));
      return;
    }

    console.log('Connecting to WebSocket:', targetUrl); // 디버깅용 로그

    isIntentionalDisconnect.current = false;
    setError(null);
    setConnectionState('connecting');

    try {
      const ws = new WebSocket(targetUrl);
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        console.log('WebSocket connected');
        setConnectionState('connected');
        setError(null);
      };

      ws.onmessage = handleMessage;

      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        console.error('Failed URL:', targetUrl);
        setError(new Error(`WebSocket connection error: ${targetUrl}`));
        setConnectionState('error');
      };

      ws.onclose = (event) => {
        console.log('WebSocket closed', {
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean,
          url: targetUrl
        });
        setConnectionState('disconnected');
        wsRef.current = null;

        // 정상 종료가 아닌 경우에만 에러 처리
        if (!event.wasClean && event.code !== 1000) {
          console.error('WebSocket closed unexpectedly', {
            code: event.code,
            reason: event.reason,
          });
        }

        // Auto reconnect (정상 종료가 아니고 의도적 종료가 아닌 경우)
        if (autoReconnect && !isIntentionalDisconnect.current && !event.wasClean) {
          console.log(`Reconnecting in ${reconnectInterval}ms...`);
          reconnectTimeoutRef.current = window.setTimeout(() => {
            // 재연결 시에도 커스텀 URL이 없으면 ref의 최신 URL 사용
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
  }, [handleMessage, autoReconnect, reconnectInterval]); // url 제거 - ref 사용

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

  // URL이 변경되면 ref 업데이트 및 기존 연결 정리
  useEffect(() => {
    if (url) {
      urlRef.current = url;
      console.log('useWebSocket URL updated to:', url);
      
      // URL이 변경되었고 기존 연결이 열려있으면 재연결
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        // 의도적인 연결 해제가 아닌 경우에만 재연결
        if (!isIntentionalDisconnect.current) {
          console.log('Closing existing connection due to URL change');
          wsRef.current.close();
        }
      }
    }
  }, [url]);

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
