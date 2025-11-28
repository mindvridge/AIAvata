/**
 * Main App Component
 *
 * Realtime AI Avatar Frontend Application
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  AvatarView,
  AudioRecorder,
  Controls,
  StatusBar,
  ConversationPanel,
  ErrorLogPanel,
} from './components';
import { useAvatarSession, useErrorLogger } from './hooks';
import type { Emotion, ConnectionState } from './types';

// Message type for conversation history
interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  emotion?: string;
}

function App() {
  // Session state
  const {
    session,
    pipelineState,
    emotion,
    isConnected,
    connectionState,
    error,
    createSession,
    sendAudio,
    setEmotion,
    disconnect,
  } = useAvatarSession({
    onVideoFrame: handleVideoFrame,
    autoConnect: false,
  });

  // Local state
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentTranscript, setCurrentTranscript] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [showConversation, setShowConversation] = useState(true);
  const [showErrorLog, setShowErrorLog] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);

  // Error logger
  const {
    logs: errorLogs,
    addError,
    addWarning,
    addInfo,
    clearLogs: clearErrorLogs,
    errorCount,
    warningCount,
  } = useErrorLogger();

  // Refs
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Handle video frames from WebSocket
  function handleVideoFrame(data: ArrayBuffer) {
    if (!canvasRef.current) return;

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    const blob = new Blob([data], { type: 'image/jpeg' });
    const url = URL.createObjectURL(blob);
    const img = new Image();

    img.onload = () => {
      ctx.drawImage(img, 0, 0, canvasRef.current!.width, canvasRef.current!.height);
      URL.revokeObjectURL(url);
    };

    img.src = url;
  }

  // Handle audio data from recorder
  const handleAudioData = useCallback((data: ArrayBuffer) => {
    if (isConnected) {
      sendAudio(data);
    }
  }, [isConnected, sendAudio]);

  // Connect handler
  const handleConnect = useCallback(async () => {
    try {
      setIsConnecting(true);
      addInfo('세션 생성 중...', 'Connection');
      await createSession({
        avatarId: 'default',
        language: 'ko',
      });
      addInfo('세션이 성공적으로 생성되었습니다. WebSocket 연결 중...', 'Connection');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '연결 실패';
      addError(`연결 실패: ${errorMessage}`, 'Connection', { error: err });
      setIsConnecting(false);
    }
  }, [createSession, addError, addInfo]);

  // WebSocket 연결 상태가 변경되면 isConnecting 업데이트
  useEffect(() => {
    if (connectionState === 'connected' || connectionState === 'error') {
      setIsConnecting(false);
    }
  }, [connectionState]);

  // Disconnect handler
  const handleDisconnect = useCallback(() => {
    setIsRecording(false);
    disconnect();
    setMessages([]);
    setCurrentTranscript('');
  }, [disconnect]);

  // Toggle mute
  const handleMuteToggle = useCallback(() => {
    setIsMuted((prev) => !prev);
  }, []);

  // Fullscreen handler
  const handleFullscreen = useCallback(() => {
    const container = document.querySelector('.avatar-container');
    if (container && document.fullscreenElement !== container) {
      container.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
  }, []);

  // Emotion change handler
  const handleEmotionChange = useCallback((newEmotion: Emotion) => {
    setEmotion(newEmotion);
  }, [setEmotion]);

  // Add message to history
  const addMessage = useCallback((role: 'user' | 'assistant', content: string, emotionStr?: string) => {
    const message: Message = {
      id: Date.now().toString(),
      role,
      content,
      timestamp: new Date(),
      emotion: emotionStr,
    };
    setMessages((prev) => [...prev, message]);
  }, []);

  return (
    <div className="min-h-screen bg-avatar-bg text-white">
      {/* Header */}
      <header className="border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center">
                <span className="text-xl">🤖</span>
              </div>
              <div>
                <h1 className="text-xl font-bold">AI Avatar</h1>
                <p className="text-xs text-gray-500">실시간 대화형 AI 아바타</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <button
                onClick={() => setShowConversation(!showConversation)}
                className="px-3 py-1.5 text-sm bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
              >
                {showConversation ? '대화 숨기기' : '대화 보기'}
              </button>
              <button
                onClick={() => setShowErrorLog(!showErrorLog)}
                className="relative px-3 py-1.5 text-sm bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors flex items-center gap-2"
              >
                <span>에러 로그</span>
                {(errorCount > 0 || warningCount > 0) && (
                  <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
                    {errorCount + warningCount}
                  </span>
                )}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="flex gap-6">
          {/* Avatar section */}
          <div className="flex-1 flex flex-col items-center gap-6">
            {/* Status bar */}
            <div className="w-full max-w-lg">
              <StatusBar
                connectionState={connectionState}
                showMetrics={isConnected}
              />
            </div>

            {/* Avatar view */}
            <div className="relative">
              <AvatarView
                emotion={emotion}
                pipelineState={pipelineState}
                isConnected={isConnected}
                isLoading={connectionState === 'connecting' || isConnecting}
                width={512}
                height={512}
              />

              {/* Hidden canvas for WebSocket frames */}
              <canvas
                ref={canvasRef}
                width={512}
                height={512}
                className="hidden"
              />
            </div>

            {/* Audio recorder */}
            <div className="mt-4">
              <AudioRecorder
                onAudioData={handleAudioData}
                isEnabled={isConnected && !isMuted}
                size="lg"
                showLevel={true}
              />
            </div>

            {/* Controls */}
            <div className="w-full max-w-lg mt-4">
              <Controls
                connectionState={connectionState}
                pipelineState={pipelineState}
                emotion={emotion}
                isMuted={isMuted}
                onConnect={handleConnect}
                onDisconnect={handleDisconnect}
                onMuteToggle={handleMuteToggle}
                onFullscreen={handleFullscreen}
                onEmotionSelect={handleEmotionChange}
                showEmotionSelector={isConnected}
              />
            </div>

            {/* Error message */}
            {error && (
              <div className="w-full max-w-lg mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg">
                <p className="text-red-400 text-sm">{error}</p>
              </div>
            )}
          </div>

          {/* Conversation panel */}
          {showConversation && (
            <div className="w-96 h-[700px]">
              <ConversationPanel
                messages={messages}
                isLoading={pipelineState === 'processing'}
                currentTranscript={currentTranscript}
              />
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 mt-auto">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between text-xs text-gray-500">
            <p>Realtime AI Avatar Service v1.0.0</p>
            <div className="flex items-center gap-4">
              <span>한국어 지원</span>
              <span>MIT License</span>
            </div>
          </div>
        </div>
      </footer>

      {/* Error Log Panel */}
      <ErrorLogPanel
        isOpen={showErrorLog}
        onClose={() => setShowErrorLog(false)}
        logs={errorLogs}
        onClear={clearErrorLogs}
      />
    </div>
  );
}

export default App;
