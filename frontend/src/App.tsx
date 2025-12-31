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
  const [isRecordingIdleVideo, setIsRecordingIdleVideo] = useState(false);
  const [hasLocalVideoFile, setHasLocalVideoFile] = useState<boolean | null>(null); // null = 확인 중

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

  // Idle 프레임 수집 상태
  const collectedFramesRef = useRef<ArrayBuffer[]>([]);
  const isCollectingRef = useRef(false);
  
  // Pipeline state와 emotion을 ref로 관리 (useAvatarSession 호출 전에 정의해야 함)
  const pipelineStateRef = useRef<string>('idle');
  const emotionRef = useRef<string>('neutral');

  // Handle video frames from WebSocket - AvatarView에 전달 + idle 프레임 수집
  // (참고: pipelineState와 emotion은 ref를 통해 접근, useAvatarSession 호출 후 업데이트됨)
  const handleVideoFrame = useCallback((data: ArrayBuffer) => {
    console.log('📥 App: Received video frame, size:', data.byteLength);
    setFrameData(data);
    
    const currentPipelineState = pipelineStateRef.current;
    const currentEmotion = emotionRef.current;
    
    // idle 상태이고 로컬 비디오가 없을 때 프레임 수집 (캐시 생성)
    if (currentPipelineState === 'idle' && !isCollectingRef.current && collectedFramesRef.current.length === 0) {
      // 먼저 캐시 확인
      import('./utils/idleVideoCache').then(({ getCachedIdleVideo }) => {
        return getCachedIdleVideo(currentEmotion);
      }).then((cached) => {
        if (!cached) {
          // 캐시가 없으면 수집 시작
          isCollectingRef.current = true;
          collectedFramesRef.current = [];
          console.log('%c🎬 Idle 프레임 수집 시작 (캐시 생성)', 'color: blue; font-weight: bold');
        }
      }).catch(() => {
        isCollectingRef.current = true;
        collectedFramesRef.current = [];
      });
    }
    
    // 프레임 수집 중이면 수집
    if (isCollectingRef.current) {
      if (collectedFramesRef.current.length < 90) {
        // 프레임 복사 (ArrayBuffer는 복사 필요)
        const frameCopy = new ArrayBuffer(data.byteLength);
        new Uint8Array(frameCopy).set(new Uint8Array(data));
        collectedFramesRef.current.push(frameCopy);
      } else {
        // 충분한 프레임 수집 완료, 비디오로 변환 후 캐시 저장
        const frames = [...collectedFramesRef.current];
        isCollectingRef.current = false;
        collectedFramesRef.current = [];
        
        // 프레임들을 비디오로 변환하고 캐시에 저장 (원본 비디오 크기: 784x1176)
        import('./utils/idleVideoCache').then(({ cacheIdleVideoFrames }) => {
          cacheIdleVideoFrames(frames, currentEmotion, 784, 1176, 30).then(() => {
            console.log(`%c✅ Idle 비디오 캐시 저장 완료: ${currentEmotion} (${frames.length} frames)`, 'color: green; font-weight: bold');
          }).catch(console.error);
        });
      }
    }
  }, []);

  // Audio context ref for playback
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioQueueRef = useRef<Array<{ data: ArrayBuffer; sampleRate: number }>>([]);
  const isPlayingAudioRef = useRef(false);

  // 오디오 재생 큐 처리
  const processAudioQueue = useCallback(async () => {
    if (isPlayingAudioRef.current || audioQueueRef.current.length === 0) {
      return;
    }

    isPlayingAudioRef.current = true;

    while (audioQueueRef.current.length > 0) {
      const { data, sampleRate } = audioQueueRef.current.shift()!;

      try {
        // AudioContext 초기화 (최초 한 번만)
        if (!audioContextRef.current) {
          audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
          console.log('%c🎵 AudioContext 초기화 완료', 'color: blue; font-weight: bold');
        }

        const audioContext = audioContextRef.current;

        // AudioContext 상태 확인 및 재개
        if (audioContext.state === 'suspended') {
          console.log('⚠️ AudioContext가 일시정지 상태입니다. 재개 시도 중...');
          try {
            await audioContext.resume();
            console.log('%c✅ AudioContext 재개 완료', 'color: green; font-weight: bold');
          } catch (err) {
            console.error('❌ AudioContext 재개 실패:', err);
            addError(`오디오 재생 불가: 브라우저가 오디오 자동 재생을 차단했습니다. 페이지를 클릭한 후 다시 시도해주세요.`, 'Audio', { error: err });
            continue;
          }
        }

        // 16-bit PCM을 Float32로 변환
        const pcmData = new Int16Array(data);
        const floatData = new Float32Array(pcmData.length);
        for (let i = 0; i < pcmData.length; i++) {
          floatData[i] = pcmData[i] / 32768.0;
        }

        // AudioBuffer 생성
        const audioBuffer = audioContext.createBuffer(1, floatData.length, sampleRate);
        audioBuffer.copyToChannel(floatData, 0);

        // 재생 (Promise로 대기하여 순차 재생 보장)
        await new Promise<void>((resolve, reject) => {
          try {
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);
            
            source.onended = () => {
              console.log(`%c✅ 오디오 재생 완료: ${(floatData.length / sampleRate).toFixed(2)}초`, 'color: green; font-weight: bold');
              resolve();
            };
            
            source.onerror = (error) => {
              console.error('%c❌ 오디오 재생 중 오류:', 'color: red; font-weight: bold', error);
              reject(error);
            };

            source.start(0);
            console.log(`%c🔊 Playing audio: ${floatData.length} samples at ${sampleRate}Hz (${(floatData.length / sampleRate).toFixed(2)}초)`, 'color: green; font-weight: bold');
          } catch (error) {
            reject(error);
          }
        });

      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : String(error);
        console.error('%c❌ 오디오 재생 실패:', 'color: red; font-weight: bold; font-size: 14px', error);
        console.error('%c오류 상세:', 'color: red; font-weight: bold', {
          error,
          dataLength: data.byteLength,
          sampleRate,
          audioContextState: audioContextRef.current?.state,
        });
        addError(`오디오 재생 실패: ${errorMsg}`, 'Audio', { 
          error: errorMsg,
          dataLength: data.byteLength,
          sampleRate,
          audioContextState: audioContextRef.current?.state,
        });
      }
    }

    isPlayingAudioRef.current = false;
  }, [addError]);

  // Handle audio data from server - AudioWaveform에 전달 + 오디오 재생
  const handleAudioDataFromServer = useCallback((data: ArrayBuffer, sampleRate: number) => {
    console.log(`%c📥 오디오 데이터 수신: ${data.byteLength} bytes, ${sampleRate}Hz`, 'color: blue; font-weight: bold');
    
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
      const errorMsg = error instanceof Error ? error.message : String(error);
      console.error('%c❌ 오디오 레벨 계산 실패:', 'color: red; font-weight: bold', error);
      addWarning(`오디오 레벨 계산 실패: ${errorMsg}`, 'Audio');
    }

    // 오디오를 큐에 추가
    audioQueueRef.current.push({ data, sampleRate });
    
    // 큐 처리 시작
    processAudioQueue();
  }, [addError, addWarning, processAudioQueue]);

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

  // pipelineState와 emotion을 ref에 동기화 (handleVideoFrame에서 사용)
  useEffect(() => {
    pipelineStateRef.current = pipelineState;
  }, [pipelineState]);

  useEffect(() => {
    emotionRef.current = emotion;
  }, [emotion]);

  // Handle audio data from recorder
  const handleAudioData = useCallback((data: ArrayBuffer) => {
    if (isConnected) {
      sendAudio(data);
    }
  }, [isConnected, sendAudio]);

  // 사용자 인터랙션 시 AudioContext 초기화 (브라우저 자동 재생 정책 대응)
  useEffect(() => {
    const initAudioContext = async () => {
      if (!audioContextRef.current) {
        try {
          audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
          // 초기 상태가 suspended일 수 있으므로 resume 시도
          if (audioContextRef.current.state === 'suspended') {
            // 사용자 인터랙션 후 자동으로 resume되도록 설정
            console.log('%c🎵 AudioContext 초기화 (suspended 상태)', 'color: blue; font-weight: bold');
          } else {
            console.log('%c🎵 AudioContext 초기화 완료', 'color: green; font-weight: bold');
          }
        } catch (error) {
          console.error('%c❌ AudioContext 초기화 실패:', 'color: red; font-weight: bold', error);
        }
      }
    };

    // 클릭 이벤트로 AudioContext 초기화
    const handleUserInteraction = () => {
      initAudioContext();
      // 한 번만 실행되도록 이벤트 제거
      document.removeEventListener('click', handleUserInteraction);
      document.removeEventListener('touchstart', handleUserInteraction);
    };

    document.addEventListener('click', handleUserInteraction);
    document.addEventListener('touchstart', handleUserInteraction);

    return () => {
      document.removeEventListener('click', handleUserInteraction);
      document.removeEventListener('touchstart', handleUserInteraction);
    };
  }, []);

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
    <div className="h-screen text-white flex flex-col overflow-hidden" style={{ backgroundColor: '#1a1a2e' }}>
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
          {/* Avatar section - 고정된 레이아웃, AudioWaveform은 절대 위치로 오버레이 */}
          <div className="flex-1 flex flex-col items-center gap-1 sm:gap-1.5 lg:gap-2 min-w-0 min-h-0 relative">
            {/* Status bar */}
            <div className="w-full max-w-full sm:max-w-lg flex-shrink-0 z-10">
              <StatusBar
                connectionState={connectionState}
                showMetrics={isConnected}
              />
            </div>

            {/* Avatar view - 원본 비디오 비율 유지 (784x1176, 세로 비디오) */}
            <div className="relative flex-shrink-0 flex flex-col items-center gap-2 sm:gap-3" style={{ maxHeight: '70vh', width: 'auto' }}>
              <div style={{ aspectRatio: '784/1176', height: '70vh', maxHeight: '70vh' }}>
                <AvatarView
                  emotion={emotion}
                  pipelineState={pipelineState}
                  isConnected={isConnected}
                  isLoading={connectionState === 'connecting' || isConnecting}
                  frameData={frameData}
                  width={784}
                  height={1176}
                />
              </div>

              {/* Audio waveform - 오른쪽 상단 작은 크기로 오버레이 */}
              <div className="absolute top-2 right-2 z-30">
                <div className="bg-gray-900/90 backdrop-blur-sm rounded-lg border border-gray-700/50 p-1.5">
                  <AudioWaveform
                    audioData={audioData}
                    audioLevel={audioLevel}
                    width={120}
                    height={30}
                    barColor="#3b82f6"
                    backgroundColor="transparent"
                    showLevel={false}
                    isActive={pipelineState === 'speaking' || pipelineState === 'processing'}
                  />
                </div>
              </div>

              {/* Audio recorder - 아바타 바로 아래 */}
              <div className="flex-shrink-0 mt-2 sm:mt-3">
                <AudioRecorder
                  onAudioData={handleAudioData}
                  isEnabled={isConnected && !isMuted}
                  size="md"
                  showLevel={false}
                />
              </div>
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
