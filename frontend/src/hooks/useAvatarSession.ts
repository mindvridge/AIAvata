/**
 * Avatar session management hook
 */

import { useState, useCallback, useRef } from 'react';
import type {
  AvatarSession,
  PipelineState,
  Emotion,
  AvatarConfig,
  UseAvatarSessionReturn,
  ConnectionState
} from '../types';
import { api } from '../services/api';
import { useWebSocket } from './useWebSocket';

interface ChatResponse {
  text: string;
  userMessage: string;
}

interface UseAvatarSessionOptions {
  onVideoFrame?: (data: ArrayBuffer) => void;
  onChatResponse?: (response: ChatResponse) => void;
  onAudioData?: (audioData: ArrayBuffer, sampleRate: number) => void;
  autoConnect?: boolean;
}

export function useAvatarSession(
  options: UseAvatarSessionOptions = {}
): UseAvatarSessionReturn {
  const { onVideoFrame, onChatResponse, onAudioData, autoConnect = false } = options;

  const [session, setSession] = useState<AvatarSession | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [wsUrl, setWsUrl] = useState<string>('');

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

  // WebSocket 메시지 핸들러 (채팅 응답 및 오디오 데이터 처리)
  const handleMessage = useCallback((message: any) => {
    if (message.type === 'chat_response') {
      onChatResponse?.({
        text: message.text,
        userMessage: message.user_message,
      });
    } else if (message.type === 'audio_data' && (message.data || message.audio)) {
      // base64로 인코딩된 오디오 데이터 디코딩
      try {
        const audioBase64 = message.data || message.audio; // "data" 또는 "audio" 필드 지원
        const binaryString = atob(audioBase64);
        const audioArrayBuffer = new ArrayBuffer(binaryString.length);
        const audioView = new Uint8Array(audioArrayBuffer);
        for (let i = 0; i < binaryString.length; i++) {
          audioView[i] = binaryString.charCodeAt(i);
        }
        onAudioData?.(audioArrayBuffer, message.sample_rate || 24000);
      } catch (error) {
        console.error('오디오 데이터 디코딩 실패:', error);
      }
    }
  }, [onChatResponse, onAudioData]);

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
    url: wsUrl || getWsUrl(), // 초기 URL 설정 (세션 생성 후 업데이트됨)
    onVideoFrame: handleVideoFrame,
    onMessage: handleMessage,
    autoReconnect: true,
  });

  const createSession = useCallback(async (config?: AvatarConfig) => {
    try {
      setError(null);

      // Create session via API
      const response = await api.createSession(config);

      // API 응답은 snake_case이므로 camelCase로 변환
      const sessionId = (response as any).session_id || response.sessionId;
      const avatarId = (response as any).avatar_id || response.avatarId;
      let websocketUrl = (response as any).websocket_url || response.websocketUrl;

      // 0.0.0.0을 localhost로 변환 (브라우저에서 0.0.0.0으로 연결할 수 없음)
      if (websocketUrl && websocketUrl.includes('0.0.0.0')) {
        websocketUrl = websocketUrl.replace('0.0.0.0', 'localhost');
      }

      // Vite 프록시가 불안정하므로 개발 환경에서도 백엔드로 직접 연결
      // 프로덕션에서는 백엔드 URL 직접 사용
      // 참고: CORS가 활성화되어 있으므로 직접 연결 가능

      console.log('Session created:', sessionId);
      console.log('Original WebSocket URL:', (response as any).websocket_url || response.websocketUrl);
      console.log('Processed WebSocket URL:', websocketUrl);

      // Update WebSocket URL with session ID
      wsUrlRef.current = websocketUrl;
      setWsUrl(websocketUrl); // URL 상태 업데이트로 useWebSocket이 새로운 URL 사용

      // Set session state
      setSession({
        sessionId: sessionId,
        avatarId: avatarId,
        currentEmotion: 'neutral',
        pipelineState: 'idle',
        totalInteractions: 0,
        createdAt: new Date().toISOString(),
        lastActivity: new Date().toISOString(),
      });

      // Connect WebSocket with the new URL directly (세션 생성 후 자동 연결)
      // 커스텀 URL을 직접 전달하여 상태 업데이트 대기 없이 즉시 연결
      console.log('Calling wsConnect with URL:', websocketUrl);
      wsConnect(websocketUrl);
    } catch (err) {
      console.error('Failed to create session:', err);
      setError(err instanceof Error ? err.message : 'Failed to create session');
    }
  }, [wsConnect]);

  const setEmotion = useCallback((newEmotion: Emotion) => {
    sendMessage({
      type: 'set_emotion',
      emotion: newEmotion,
    } as { type: string; emotion: Emotion });
  }, [sendMessage]);

  // 텍스트 채팅 메시지 전송
  const sendChat = useCallback((text: string) => {
    if (!text.trim()) return;
    
    sendMessage({
      type: 'chat',
      text: text.trim(),
    } as { type: string; text: string });
  }, [sendMessage]);

  const disconnect = useCallback(() => {
    wsDisconnect();

    // Delete session via API
    if (session?.sessionId) {
      api.deleteSession(session.sessionId).catch(console.error);
    }

    setSession(null);
    wsUrlRef.current = '';
    setWsUrl(''); // URL 상태 초기화
  }, [wsDisconnect, session]);

  return {
    session,
    pipelineState,
    emotion,
    isConnected,
    connectionState,
    error,
    createSession,
    sendAudio,
    sendChat,
    setEmotion,
    disconnect,
  };
}

export default useAvatarSession;
