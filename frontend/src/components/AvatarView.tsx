/**
 * AvatarView Component
 *
 * Displays the avatar video/canvas with emotion indicators and loading states
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { User, Loader2 } from 'lucide-react';
import { clsx } from 'clsx';
import type { Emotion, PipelineState } from '../types';

interface AvatarViewProps {
  emotion?: Emotion;
  pipelineState?: PipelineState;
  isConnected?: boolean;
  isLoading?: boolean;
  videoTrack?: MediaStreamTrack | null;
  frameData?: ArrayBuffer | null;  // WebSocket에서 받은 비디오 프레임 데이터
  onFrameData?: (data: ArrayBuffer) => void;
  onRecordingStateChange?: (isRecording: boolean) => void; // 녹화 상태 변경 콜백 (선택적)
  width?: number;
  height?: number;
}

const EMOTION_LABELS: Record<Emotion, string> = {
  neutral: '중립',
  happy: '기쁨',
  sad: '슬픔',
  angry: '화남',
  surprised: '놀람',
  listening: '듣는 중',
  thinking: '생각 중',
  sympathetic: '공감',
  concerned: '걱정',
};

const PIPELINE_LABELS: Record<PipelineState, string> = {
  idle: '대기 중',
  listening: '듣는 중...',
  processing: '처리 중...',
  speaking: '말하는 중...',
  error: '오류',
};

export function AvatarView({
  emotion = 'neutral',
  pipelineState = 'idle',
  isConnected = false,
  isLoading = false,
  videoTrack = null,
  frameData = null,
  onFrameData,
  onRecordingStateChange,
  width: initialWidth = 784,
  height: initialHeight = 1176,
}: AvatarViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const localVideoRef = useRef<HTMLVideoElement>(null); // 로컬 idle 루프 비디오
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [useCanvas, setUseCanvas] = useState(!videoTrack);
  const canvasInitialized = useRef(false);
  const [useLocalVideo, setUseLocalVideo] = useState(false); // 로컬 비디오 사용 여부
  // 🔑 동적 Canvas 크기 (첫 프레임에서 자동 감지, 기본값: 784x1176)
  const [canvasSize, setCanvasSize] = useState({ width: initialWidth, height: initialHeight });
  const { width, height } = canvasSize;

  // Attach video track to video element
  useEffect(() => {
    if (videoTrack && videoRef.current) {
      const stream = new MediaStream([videoTrack]);
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(console.error);
      setUseCanvas(false);
      setUseLocalVideo(false);
    } else {
      setUseCanvas(true);
    }
  }, [videoTrack]);

  // 로컬 비디오 재생 관리: idle 상태일 때만 재생, speaking/processing 상태일 때 일시정지
  useEffect(() => {
    if (!localVideoRef.current || !isConnected) return;

    const localVideo = localVideoRef.current;
    const currentEmotion = emotion || 'neutral';
    const emotionLabel = EMOTION_LABELS[currentEmotion] || currentEmotion;
    const videoUrl = `/avatars/avata_ani.mp4`; // avata_ani.mp4만 사용
    
    // WebSocket 프레임이 없을 때 로컬 비디오 재생 (idle, processing, listening 등)
    // speaking 상태일 때는 TTS 오디오와 함께 프레임이 올 수 있으므로 프레임이 없을 때만 재생
    const shouldPlayLocal = !frameData && (pipelineState === 'idle' || pipelineState === 'processing' || pipelineState === 'listening');
    
    if (shouldPlayLocal && !useLocalVideo) {
      console.log(`%c🎬 로컬 비디오 재생 시작 시도: ${emotionLabel}`, 'color: blue; font-weight: bold');
      
      // 먼저 캐시된 idle 비디오 확인
      import('../utils/idleVideoCache').then(({ getCachedIdleVideo }) => {
        return getCachedIdleVideo(currentEmotion);
      }).then((cachedVideoBlob) => {
        if (cachedVideoBlob) {
          // 캐시된 비디오 파일 재생
          const blobUrl = URL.createObjectURL(cachedVideoBlob);
          localVideo.src = blobUrl;
          localVideo.loop = true;
          localVideo.muted = true;
          return localVideo.play().then(() => {
            setUseLocalVideo(true); // 재생 성공 후 상태 업데이트
            console.log(`%c✅ 캐시된 idle 비디오 재생 시작: ${emotionLabel}`, 'color: green; font-weight: bold');
          }).catch((err) => {
            console.error(`%c❌ 캐시된 비디오 재생 실패:`, 'color: red; font-weight: bold', err);
            URL.revokeObjectURL(blobUrl);
          });
        } else {
          // 캐시가 없으면 파일 확인 후 재생 시도
          console.log(`%c📁 로컬 파일 확인 중: ${videoUrl}`, 'color: blue; font-weight: bold');
          return fetch(videoUrl, { method: 'HEAD' })
            .then((response) => {
              if (response.ok) {
                // 파일이 있으면 직접 재생
                localVideo.src = videoUrl;
                localVideo.loop = true;
                localVideo.muted = true;
                return localVideo.play().then(() => {
                  setUseLocalVideo(true); // 재생 성공 후 상태 업데이트
                  console.log(`%c✅ 로컬 idle 비디오 재생 시작: ${emotionLabel}`, 'color: green; font-weight: bold');
                }).catch((err) => {
                  console.error(`%c❌ 로컬 비디오 재생 실패:`, 'color: red; font-weight: bold', err);
                });
              } else {
                // 파일이 없으면 서버 스트림 사용 (프레임 수집 시작)
                console.log(`%c⚠️ 로컬 비디오 파일 없음: ${videoUrl}, 서버 스트림에서 캐시 생성`, 'color: orange; font-weight: bold');
                throw new Error('Local video file not found');
              }
            });
        }
      }).catch((error) => {
        console.warn(`%c⚠️ 비디오 로드 실패 (서버 스트림 사용): ${error.message}`, 'color: orange; font-weight: bold');
      });
    } else if (!shouldPlayLocal && useLocalVideo) {
      // speaking/processing 상태이거나 WebSocket 프레임이 있을 때 일시정지
      localVideo.pause();
      // Blob URL 정리 (메모리 누수 방지)
      if (localVideo.src && localVideo.src.startsWith('blob:')) {
        URL.revokeObjectURL(localVideo.src);
      }
      setUseLocalVideo(false);
      console.log('%c⏸️ 로컬 idle 비디오 일시정지 (WebSocket 스트림 활성화)', 'color: orange; font-weight: bold');
    }

    // cleanup: 컴포넌트 언마운트 시 Blob URL 정리
    return () => {
      if (localVideoRef.current?.src && localVideoRef.current.src.startsWith('blob:')) {
        URL.revokeObjectURL(localVideoRef.current.src);
      }
    };
  }, [pipelineState, frameData, isConnected, useLocalVideo, emotion]);

  // Handle frame data from WebSocket
  const handleFrameData = useCallback((data: ArrayBuffer) => {
    if (!canvasRef.current || !useCanvas) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Create blob and image from frame data
    const blob = new Blob([data], { type: 'image/jpeg' });
    const url = URL.createObjectURL(blob);
    const img = new Image();

    img.onload = () => {
      // 🔑 원본 이미지 크기로 그리기 (찌그러짐 방지)
      const imgWidth = img.naturalWidth;
      const imgHeight = img.naturalHeight;
      if (imgWidth > 0 && imgHeight > 0 && (canvas.width !== imgWidth || canvas.height !== imgHeight)) {
        setCanvasSize({ width: imgWidth, height: imgHeight });
        canvas.width = imgWidth;
        canvas.height = imgHeight;
      }
      ctx.drawImage(img, 0, 0, imgWidth, imgHeight);
      URL.revokeObjectURL(url);
    };

    img.src = url;
  }, [useCanvas]);

  // Expose frame handler
  useEffect(() => {
    if (onFrameData) {
      // This is handled by parent component
    }
  }, [onFrameData]);

  // Canvas 초기화는 frameData가 있을 때만 수행 (WebSocket 프레임용)

  // 로딩 표시는 로컬 비디오나 배경으로 대체되므로 제거 (로컬 비디오가 표시됨)

  // frameData prop이 변경되면 canvas에 그리기
  // 프레임 카운터 (디버깅용)
  const frameCountRef = useRef(0);


  useEffect(() => {
    // WebSocket 프레임이 오면 로컬 비디오 일시정지
    if (frameData && useLocalVideo && localVideoRef.current) {
      localVideoRef.current.pause();
      setUseLocalVideo(false);
    }

    if (!frameData || !canvasRef.current || !useCanvas) {
      return;
    }

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) {
      console.error('🖼️ Failed to get 2D context');
      return;
    }

    // Canvas 배경 초기화 (처음 프레임일 때만)
    if (!canvasInitialized.current) {
      ctx.fillStyle = '#111827'; // gray-900
      ctx.fillRect(0, 0, width, height);
      canvasInitialized.current = true;
    }

    frameCountRef.current++;
    // 처음 3개 프레임과 이후 30개마다 로깅
    if (frameCountRef.current <= 3 || frameCountRef.current % 30 === 0) {
      console.log(`🖼️ Drawing frame #${frameCountRef.current} to canvas, size: ${frameData.byteLength} bytes`);
    }

    const blob = new Blob([frameData], { type: 'image/jpeg' });
    const url = URL.createObjectURL(blob);
    const img = new Image();

    img.onload = () => {
      // Get fresh context to ensure it's still valid
      const canvas = canvasRef.current;
      if (!canvas) {
        console.warn('🖼️ Canvas ref lost before frame could be drawn');
        URL.revokeObjectURL(url);
        return;
      }

      // 🔑 첫 프레임에서 Canvas 크기를 이미지 크기에 맞게 자동 조정 (찌그러짐 방지)
      const imgWidth = img.naturalWidth;
      const imgHeight = img.naturalHeight;
      
      // 🔑 img 객체 정보 콘솔 출력 (디버깅용)
      console.log('🖼️ Image 객체 정보:', {
        img: img,
        naturalWidth: img.naturalWidth,
        naturalHeight: img.naturalHeight,
        width: img.width,
        height: img.height,
        src: img.src,
        complete: img.complete,
        blobUrl: url,
        frameNumber: frameCountRef.current
      });
      
      // 이미지 크기가 유효한지 확인 (서버에서 전송된 실제 크기 사용)
      if (imgWidth > 0 && imgHeight > 0) {
        // Canvas 크기가 이미지 크기와 다르면 조정
        if (canvas.width !== imgWidth || canvas.height !== imgHeight) {
          console.log(`🖼️ Canvas 크기 자동 조정: ${canvas.width}x${canvas.height} → ${imgWidth}x${imgHeight} (이미지 실제 크기)`);
          setCanvasSize({ width: imgWidth, height: imgHeight });
          canvas.width = imgWidth;
          canvas.height = imgHeight;
        }
      } else {
        // 이미지 크기를 읽을 수 없으면 초기값 유지
        console.warn(`🖼️ 이미지 크기를 읽을 수 없음: naturalWidth=${imgWidth}, naturalHeight=${imgHeight}, 초기 크기 유지: ${canvas.width}x${canvas.height}`);
      }

      const freshCtx = canvas.getContext('2d');
      if (!freshCtx) {
        console.warn('🖼️ Could not get canvas context in onload');
        URL.revokeObjectURL(url);
        return;
      }
      // 🔑 원본 이미지 크기로 그리기 (찌그러짐 방지)
      freshCtx.drawImage(img, 0, 0, imgWidth, imgHeight);
      if (frameCountRef.current <= 3) {
        console.log(`🖼️ Frame #${frameCountRef.current} drawn successfully (${imgWidth}x${imgHeight})`);
      }
      URL.revokeObjectURL(url);
    };

    img.onerror = (error) => {
      console.error('🖼️ Failed to load frame image:', error);
      URL.revokeObjectURL(url);
    };

    img.src = url;
  }, [frameData, useCanvas, width, height]);

  // Draw placeholder background when not connected (텍스트는 HTML overlay에서 처리)
  useEffect(() => {
    if (!useCanvas || !canvasRef.current || isConnected) return;

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    // Draw gradient background only (텍스트는 제거하여 HTML overlay와 겹침 방지)
    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, '#16213e');
    gradient.addColorStop(1, '#0f3460');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
  }, [useCanvas, isConnected, width, height]);

  return (
    <div className="avatar-container relative" style={{ width: '100%', height: '100%' }}>
      {/* Video element (for LiveKit) */}
      {!useCanvas && (
        <video
          ref={videoRef}
          className="avatar-video w-full h-full"
          autoPlay
          playsInline
          muted
          style={{ display: videoTrack ? 'block' : 'none' }}
        />
      )}

      {/* Canvas element - 항상 렌더링 (WebSocket 프레임 또는 배경용) */}
      {useCanvas && (
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="avatar-video w-full h-full"
          style={{ 
            backgroundColor: '#111827',
            display: frameData ? 'block' : (useLocalVideo ? 'none' : 'block'), // 비디오가 재생 중이면 Canvas 숨김
            objectFit: 'contain',
            position: 'absolute',
            top: 0,
            left: 0,
            zIndex: frameData ? 2 : 0,
          }}
        />
      )}

      {/* Local idle video (idle 상태일 때 재생) - WebSocket 프레임이 없을 때만 표시 */}
      {useCanvas && !frameData && (
        <video
          ref={localVideoRef}
          className="avatar-video w-full h-full"
          loop
          muted
          playsInline
          style={{ 
            display: useLocalVideo ? 'block' : 'none',
            objectFit: 'contain',
            position: 'absolute',
            top: 0,
            left: 0,
            zIndex: 1,
            backgroundColor: '#111827',
          }}
        />
      )}


      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50">
          <div className="flex flex-col items-center gap-3">
            <Loader2 className="w-12 h-12 text-primary-400 animate-spin" />
            <span className="text-white text-sm">연결 중...</span>
          </div>
        </div>
      )}

      {/* Placeholder when not connected */}
      {!isConnected && !isLoading && (
        <div className="absolute inset-0 flex items-center justify-center z-10">
          <div className="flex flex-col items-center gap-4 text-gray-400">
            <div className="w-32 h-32 rounded-full bg-gray-700/50 flex items-center justify-center">
              <User className="w-16 h-16" />
            </div>
            <div className="flex flex-col items-center gap-1">
              <p className="text-sm font-medium">아바타 대기 화면</p>
              <p className="text-xs text-gray-500">"시작" 버튼을 누르세요</p>
            </div>
          </div>
        </div>
      )}

      {/* Status overlay */}
      {isConnected && (
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/70 to-transparent">
          <div className="flex items-center justify-between">
            {/* Emotion badge */}
            <span className={clsx('emotion-badge', emotion)}>
              {EMOTION_LABELS[emotion]}
            </span>

            {/* Pipeline state */}
            <div className="flex items-center gap-2">
              {pipelineState === 'processing' && (
                <Loader2 className="w-4 h-4 text-primary-400 animate-spin" />
              )}
              <span className="text-white/80 text-sm">
                {PIPELINE_LABELS[pipelineState]}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Speaking indicator */}
      {pipelineState === 'speaking' && (
        <div className="absolute top-4 right-4">
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span className="w-2 h-3 bg-green-500 rounded-full animate-pulse delay-75" />
            <span className="w-2 h-4 bg-green-500 rounded-full animate-pulse delay-150" />
            <span className="w-2 h-3 bg-green-500 rounded-full animate-pulse delay-75" />
            <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          </div>
        </div>
      )}

      {/* Listening indicator */}
      {pipelineState === 'listening' && (
        <div className="absolute top-4 right-4">
          <div className="w-8 h-8 rounded-full border-2 border-primary-400 animate-pulse-ring" />
        </div>
      )}
    </div>
  );
}

export default AvatarView;
