/**
 * Avatar session management hook
 */

import { useState, useCallback, useRef } from 'react';
import type {
  AvatarSession,
  PipelineState,
  Emotion,
  AvatarConfig,
  UseAvatarSessionReturn
} from '../types';
import { api } from '../lib/api';
import { useWebSocket } from './useWebSocket';

interface UseAvatarSessionOptions {
  onVideoFrame?: (data: ArrayBuffer) => void;
  autoConnect?: boolean;
}

export function useAvatarSession(
  options: UseAvatarSessionOptions = {}
): UseAvatarSessionReturn {
  const { onVideoFrame, autoConnect = false } = options;

  const [session, setSession] = useState<AvatarSession | null>(null);
  const [error, setError] = useState<string | null>(null);

  const wsUrlRef = useRef<string>('');

  // Determine WebSocket URL
  const getWsUrl = useCallback(() => {
    if (wsUrlRef.current) return wsUrlRef.current;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/ws/avatar`;
  }, []);

  const handleVideoFrame = useCallback((data: ArrayBuffer) => {
    onVideoFrame?.(data);
  }, [onVideoFrame]);

  const {
    isConnected,
    connectionState,
    pipelineState,
    emotion,
    connect: wsConnect,
    disconnect: wsDisconnect,
    sendAudio,
    sendMessage,
  } = useWebSocket({
    url: getWsUrl(),
    onVideoFrame: handleVideoFrame,
    autoReconnect: true,
  });

  const createSession = useCallback(async (config?: AvatarConfig) => {
    try {
      setError(null);

      // Create session via API
      const response = await api.createSession(config);

      // Update WebSocket URL with session ID
      wsUrlRef.current = response.websocketUrl;

      // Set session state
      setSession({
        sessionId: response.sessionId,
        avatarId: response.avatarId,
        currentEmotion: 'neutral',
        pipelineState: 'idle',
        totalInteractions: 0,
        createdAt: new Date().toISOString(),
        lastActivity: new Date().toISOString(),
      });

      // Connect WebSocket
      if (autoConnect) {
        wsConnect();
      }

      console.log('Session created:', response.sessionId);
    } catch (err) {
      console.error('Failed to create session:', err);
      setError(err instanceof Error ? err.message : 'Failed to create session');
    }
  }, [autoConnect, wsConnect]);

  const setEmotion = useCallback((newEmotion: Emotion) => {
    sendMessage({
      type: 'set_emotion',
      emotion: newEmotion,
    } as { type: string; emotion: Emotion });
  }, [sendMessage]);

  const disconnect = useCallback(() => {
    wsDisconnect();

    // Delete session via API
    if (session?.sessionId) {
      api.deleteSession(session.sessionId).catch(console.error);
    }

    setSession(null);
    wsUrlRef.current = '';
  }, [wsDisconnect, session]);

  return {
    session,
    pipelineState,
    emotion,
    isConnected,
    error,
    createSession,
    sendAudio,
    setEmotion,
    disconnect,
  };
}

export default useAvatarSession;
