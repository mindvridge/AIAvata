/**
 * Main App Component
 *
 * Realtime AI Avatar Frontend Application
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  AvatarView,
  AudioRecorder,
  AudioWaveform,
  Controls,
  StatusBar,
  ConversationPanel,
  ChatInput,
  ErrorLogPanel,
  VoiceManagement,
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
  // Local state (useAvatarSession보다 먼저 정의)
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentTranscript, setCurrentTranscript] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [showConversation, setShowConversation] = useState(true);
  const [showErrorLog, setShowErrorLog] = useState(false);
  const [showVoiceManagement, setShowVoiceManagement] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [frameData, setFrameData] = useState<ArrayBuffer | null>(null);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [audioData, setAudioData] = useState<ArrayBuffer | null>(null);
  const [audioLevel, setAudioLevel] = useState(0);

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

  // Handle video frames from WebSocket - AvatarView에 전달
  const handleVideoFrame = useCallback((data: ArrayBuffer) => {
    console.debug('App: Received video frame, size:', data.byteLength);
    setFrameData(data);
  }, []);

  // Audio context ref for playback
  const audioContextRef = useRef<AudioContext | null>(null);

  // Handle audio data from server - AudioWaveform에 전달 + 오디오 재생
  const handleAudioDataFromServer = useCallback((data: ArrayBuffer, sampleRate: number) => {
    setAudioData(data);

    // 오디오 레벨 계산 (간단한 방식)
    try {
      const audioView = new Int16Array(data);
      let sum = 0;
      for (let i = 0; i < audioView.length; i++) {
        sum += Math.abs(audioView[i]);
      }
      const average = sum / audioView.length;
      const level = Math.min(average / 32767, 1.0); // 0-1 범위로 정규화
      setAudioLevel(level);
    } catch (error) {
      console.error('오디오 레벨 계산 실패:', error);
    }

    // 오디오 재생
    try {
      // AudioContext 초기화 (최초 한 번만)
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      }

      const audioContext = audioContextRef.current;

      // 16-bit PCM을 Float32로 변환
      const pcmData = new Int16Array(data);
      const floatData = new Float32Array(pcmData.length);
      for (let i = 0; i < pcmData.length; i++) {
        floatData[i] = pcmData[i] / 32768.0;
      }

      // AudioBuffer 생성
      const audioBuffer = audioContext.createBuffer(1, floatData.length, sampleRate);
      audioBuffer.copyToChannel(floatData, 0);

      // 재생
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContext.destination);
      source.start(0);

      console.log(`🔊 Playing audio: ${floatData.length} samples at ${sampleRate}Hz`);
    } catch (error) {
      console.error('오디오 재생 실패:', error);
    }
  }, []);

  // Handle chat response from WebSocket
  const handleChatResponse = useCallback((response: { text: string; userMessage: string }) => {
    const assistantMessage: Message = {
      id: `assistant-${Date.now()}`,
      role: 'assistant',
      content: response.text,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, assistantMessage]);
    setIsChatLoading(false);
  }, []); // setMessages와 setIsChatLoading은 setState 함수이므로 의존성 배열에 필요 없음

  // Session state (함수들이 모두 정의된 후에 호출)
  const {
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
  } = useAvatarSession({
    onVideoFrame: handleVideoFrame,
    onChatResponse: handleChatResponse,
    onAudioData: handleAudioDataFromServer,
    autoConnect: false,
  });

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

  // Chat message handler
  const handleChatSend = useCallback((text: string) => {
    if (!isConnected || !text.trim()) return;

    // 사용자 메시지 추가
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMessage]);
    setIsChatLoading(true);

    // WebSocket으로 채팅 메시지 전송
    sendChat(text);
    
    // 응답은 WebSocket onChatResponse callback에서 처리됨
  }, [isConnected, sendChat]);

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
    <div className="h-screen bg-avatar-bg text-white flex flex-col overflow-hidden">
      {/* Header */}
      <header className="border-b border-gray-800 flex-shrink-0">
        <div className="max-w-[1920px] mx-auto px-2 sm:px-4 py-1.5 sm:py-2">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-2 sm:gap-3">
              <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center flex-shrink-0">
                <span className="text-base sm:text-xl">🤖</span>
              </div>
              <div>
                <h1 className="text-base sm:text-xl font-bold">AI Avatar</h1>
                <p className="text-xs text-gray-500 hidden sm:block">실시간 대화형 AI 아바타</p>
              </div>
            </div>

            <div className="flex items-center gap-2 sm:gap-4">
              <button
                onClick={() => setShowVoiceManagement(true)}
                className="px-2 sm:px-3 py-1 sm:py-1.5 text-xs sm:text-sm bg-purple-700 hover:bg-purple-600 rounded-lg transition-colors flex items-center gap-1"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
                <span className="hidden sm:inline">Voice</span>
              </button>
              <button
                onClick={() => setShowConversation(!showConversation)}
                className="px-2 sm:px-3 py-1 sm:py-1.5 text-xs sm:text-sm bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
              >
                <span className="hidden sm:inline">{showConversation ? '대화 숨기기' : '대화 보기'}</span>
                <span className="sm:hidden">{showConversation ? '숨기기' : '대화'}</span>
              </button>
              <button
                onClick={() => setShowErrorLog(!showErrorLog)}
                className="relative px-2 sm:px-3 py-1 sm:py-1.5 text-xs sm:text-sm bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors flex items-center gap-2"
              >
                <span className="hidden sm:inline">에러 로그</span>
                <span className="sm:hidden">로그</span>
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

      {/* Main content - flex로 공간 최적화, 스크롤 없음 */}
      <main className="flex-1 overflow-hidden max-w-[1920px] mx-auto w-full px-2 sm:px-4 py-1 sm:py-2 min-h-0">
        <div className="flex flex-col lg:flex-row gap-2 sm:gap-3 h-full">
          {/* Avatar section - 컴팩트하게 배치, 공간 균등 분배 */}
          <div className="flex-1 flex flex-col items-center gap-1 sm:gap-1.5 lg:gap-2 min-w-0 min-h-0 justify-center">
            {/* Status bar */}
            <div className="w-full max-w-full sm:max-w-lg flex-shrink-0">
              <StatusBar
                connectionState={connectionState}
                showMetrics={isConnected}
              />
            </div>

            {/* Avatar view - 화면에 맞게 크기 조정, 남은 공간 활용 */}
            <div className="relative w-full max-w-full sm:max-w-md lg:max-w-lg flex-[2] min-h-0 flex items-center justify-center">
              <div className="w-full h-full max-h-full aspect-square max-w-full">
                <AvatarView
                  emotion={emotion}
                  pipelineState={pipelineState}
                  isConnected={isConnected}
                  isLoading={connectionState === 'connecting' || isConnecting}
                  frameData={frameData}
                  width={512}
                  height={512}
                />
              </div>
            </div>

            {/* Audio waveform - 작게 */}
            <div className="w-full max-w-full sm:max-w-lg flex-shrink-0">
              <AudioWaveform
                audioData={audioData}
                audioLevel={audioLevel}
                width={512}
                height={60}
                barColor="#3b82f6"
                backgroundColor="#1f2937"
                showLevel={false}
                isActive={pipelineState === 'speaking' || pipelineState === 'processing'}
              />
            </div>

            {/* Audio recorder - 작게 */}
            <div className="w-full max-w-full sm:max-w-lg flex-shrink-0">
              <AudioRecorder
                onAudioData={handleAudioData}
                isEnabled={isConnected && !isMuted}
                size="md"
                showLevel={false}
              />
            </div>

            {/* Chat input - 작게 */}
            <div className="w-full max-w-full sm:max-w-lg flex-shrink-0">
              <ChatInput
                onSend={handleChatSend}
                isLoading={isChatLoading}
                isDisabled={!isConnected}
                placeholder="메시지 입력... (Enter)"
              />
            </div>

            {/* Controls - 컴팩트 */}
            <div className="w-full max-w-full sm:max-w-lg flex-shrink-0">
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
              <div className="w-full max-w-full sm:max-w-lg p-2 sm:p-3 bg-red-500/20 border border-red-500/50 rounded-lg flex-shrink-0">
                <p className="text-red-400 text-xs">{error}</p>
              </div>
            )}
          </div>

          {/* Conversation panel - 전체 높이 사용 */}
          {showConversation && (
            <div className="w-full lg:w-80 xl:w-96 flex-shrink-0 h-full min-h-0">
              <ConversationPanel
                messages={messages}
                isLoading={pipelineState === 'processing'}
                currentTranscript={currentTranscript}
              />
            </div>
          )}
        </div>
      </main>

      {/* Footer - 최소화 */}
      <footer className="border-t border-gray-800 flex-shrink-0">
        <div className="max-w-[1920px] mx-auto px-2 sm:px-4 py-1">
          <div className="flex items-center justify-between text-xs text-gray-500 flex-wrap gap-1">
            <p className="text-xs truncate">AI Avatar v1.0.0</p>
            <div className="flex items-center gap-2 text-xs">
              <span className="hidden sm:inline">한국어</span>
              <span className="hidden md:inline">MIT</span>
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

      {/* Voice Management Panel */}
      {showVoiceManagement && (
        <VoiceManagement
          onClose={() => setShowVoiceManagement(false)}
          onSelectVoice={(voiceId) => {
            console.log('Selected voice:', voiceId);
            addInfo(`Voice selected: ${voiceId}`, 'Voice');
            setShowVoiceManagement(false);
          }}
        />
      )}
    </div>
  );
}

export default App;
