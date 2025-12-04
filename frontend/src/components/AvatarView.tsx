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
  width = 512,
  height = 512,
}: AvatarViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [useCanvas, setUseCanvas] = useState(!videoTrack);
  const canvasInitialized = useRef(false);

  // Attach video track to video element
  useEffect(() => {
    if (videoTrack && videoRef.current) {
      const stream = new MediaStream([videoTrack]);
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(console.error);
      setUseCanvas(false);
    } else {
      setUseCanvas(true);
    }
  }, [videoTrack]);

  // Handle frame data from WebSocket
  const handleFrameData = useCallback((data: ArrayBuffer) => {
    if (!canvasRef.current || !useCanvas) return;

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    // Create blob and image from frame data
    const blob = new Blob([data], { type: 'image/jpeg' });
    const url = URL.createObjectURL(blob);
    const img = new Image();

    img.onload = () => {
      ctx.drawImage(img, 0, 0, width, height);
      URL.revokeObjectURL(url);
    };

    img.src = url;
  }, [useCanvas, width, height]);

  // Expose frame handler
  useEffect(() => {
    if (onFrameData) {
      // This is handled by parent component
    }
  }, [onFrameData]);

  // Canvas 초기화 및 배경 그리기 (한 번만 실행)
  useEffect(() => {
    if (!useCanvas || !canvasRef.current || canvasInitialized.current) return;

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    // Canvas 전체를 초기 배경색으로 채우기
    ctx.fillStyle = '#111827'; // gray-900
    ctx.fillRect(0, 0, width, height);
    
    canvasInitialized.current = true;
    console.debug('Canvas initialized with background');
  }, [useCanvas, width, height]);

  // 연결 후 프레임이 없을 때 로딩 표시
  useEffect(() => {
    if (!useCanvas || !canvasRef.current || !isConnected || frameData) return;

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    // 로딩 배경
    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, '#1a1a2e');
    gradient.addColorStop(1, '#16213e');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    // 로딩 텍스트
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.font = '18px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('프레임 대기 중...', width / 2, height / 2);
  }, [useCanvas, isConnected, frameData, width, height]);

  // frameData prop이 변경되면 canvas에 그리기
  // 프레임 카운터 (디버깅용)
  const frameCountRef = useRef(0);

  useEffect(() => {
    if (!frameData || !canvasRef.current || !useCanvas) {
      if (!frameData) console.log('🖼️ No frameData');
      if (!canvasRef.current) console.log('🖼️ No canvas ref');
      if (!useCanvas) console.log('🖼️ useCanvas is false');
      return;
    }

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) {
      console.error('🖼️ Failed to get 2D context');
      return;
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
      const freshCtx = canvas.getContext('2d');
      if (!freshCtx) {
        console.warn('🖼️ Could not get canvas context in onload');
        URL.revokeObjectURL(url);
        return;
      }
      freshCtx.drawImage(img, 0, 0, width, height);
      if (frameCountRef.current <= 3) {
        console.log(`🖼️ Frame #${frameCountRef.current} drawn successfully (${img.naturalWidth}x${img.naturalHeight})`);
      }
      URL.revokeObjectURL(url);
    };

    img.onerror = (error) => {
      console.error('🖼️ Failed to load frame image:', error);
      URL.revokeObjectURL(url);
    };

    img.src = url;
  }, [frameData, useCanvas, width, height]);

  // Draw placeholder when not connected
  useEffect(() => {
    if (!useCanvas || !canvasRef.current || isConnected) return;

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    // Draw gradient background
    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, '#16213e');
    gradient.addColorStop(1, '#0f3460');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    // Draw placeholder icon
    ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.beginPath();
    ctx.arc(width / 2, height / 2 - 30, 80, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.font = '16px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('아바타에 연결하려면', width / 2, height / 2 + 80);
    ctx.fillText('"시작" 버튼을 누르세요', width / 2, height / 2 + 105);
  }, [useCanvas, isConnected, width, height]);

  return (
    <div className="avatar-container relative w-full h-full">
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

      {/* Canvas element (for WebSocket frames) */}
      {useCanvas && (
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="avatar-video w-full h-full"
          style={{ 
            backgroundColor: '#111827',
            display: 'block',
            objectFit: 'contain',
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
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="flex flex-col items-center gap-4 text-gray-400">
            <div className="w-32 h-32 rounded-full bg-gray-700/50 flex items-center justify-center">
              <User className="w-16 h-16" />
            </div>
            <p className="text-sm">아바타 대기 중</p>
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
