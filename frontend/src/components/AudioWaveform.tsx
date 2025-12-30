/**
 * AudioWaveform Component
 * 
 * 아바타 음성 파동 그래프 시각화 컴포넌트
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { clsx } from 'clsx';

interface AudioWaveformProps {
  audioData?: ArrayBuffer | null;
  audioLevel?: number; // 0-1 범위의 오디오 레벨
  width?: number;
  height?: number;
  barColor?: string;
  backgroundColor?: string;
  showLevel?: boolean;
  isActive?: boolean; // 아바타가 말하는 중인지
}

export function AudioWaveform({
  audioData,
  audioLevel = 0,
  width = 400,
  height = 100,
  barColor = '#3b82f6',
  backgroundColor = '#1f2937',
  showLevel = true,
  isActive = false,
}: AudioWaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const [waveformData, setWaveformData] = useState<number[]>([]);

  // AudioContext 및 AnalyserNode 초기화
  useEffect(() => {
    if (!audioContextRef.current) {
      try {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
        analyserRef.current = audioContextRef.current.createAnalyser();
        analyserRef.current.fftSize = 256;
        analyserRef.current.smoothingTimeConstant = 0.8;
        dataArrayRef.current = new Uint8Array(analyserRef.current.frequencyBinCount);
      } catch (e) {
        console.error('AudioContext 초기화 실패:', e);
      }
    }
  }, []);

  // 오디오 데이터로부터 파형 생성
  const processAudioData = useCallback(async (data: ArrayBuffer) => {
    try {
      // 백엔드에서 전송하는 오디오는 PCM raw bytes (int16, 24kHz)
      // WAV 헤더 없이 raw PCM 데이터를 직접 처리
      const audioView = new Int16Array(data);
      const samples = Math.min(audioView.length, width * 2); // 너비에 맞춤 (오버샘플링)
      const step = Math.floor(audioView.length / samples);
      
      const waveform: number[] = [];
      for (let i = 0; i < samples; i++) {
        const index = i * step;
        const sample = Math.abs(audioView[index]) / 32767; // -1~1 범위로 정규화
        waveform.push(sample);
      }
      
      setWaveformData(waveform);

      // AudioContext가 있으면 실시간 분석도 가능하도록 설정
      if (audioContextRef.current && analyserRef.current) {
        // PCM 데이터를 Float32Array로 변환
        const float32Data = new Float32Array(audioView.length);
        for (let i = 0; i < audioView.length; i++) {
          float32Data[i] = audioView[i] / 32767.0;
        }
        
        // AudioBuffer 생성 및 분석
        const audioBuffer = audioContextRef.current.createBuffer(
          1, // 채널 수 (모노)
          float32Data.length,
          24000 // 샘플레이트 (TTS 기본값)
        );
        audioBuffer.copyToChannel(float32Data, 0);
        
        const source = audioContextRef.current.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(analyserRef.current);
        source.start(0);
      }
    } catch (e) {
      console.error('오디오 데이터 처리 실패:', e);
    }
  }, [width]);

  // 오디오 데이터가 변경되면 처리
  useEffect(() => {
    if (audioData) {
      processAudioData(audioData);
    }
  }, [audioData, processAudioData]);

  // 파형 그리기
  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 캔버스 초기화
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);

    // 중앙선 그리기
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // 파형 그리기
    if (waveformData.length > 0) {
      ctx.strokeStyle = barColor;
      ctx.lineWidth = 2;
      ctx.beginPath();

      const barWidth = width / waveformData.length;

      for (let i = 0; i < waveformData.length; i++) {
        const x = i * barWidth;
        const amplitude = waveformData[i] * height * 0.4; // 최대 높이의 40%
        
        ctx.moveTo(x, height / 2 - amplitude);
        ctx.lineTo(x, height / 2 + amplitude);
      }

      ctx.stroke();
    } else if (analyserRef.current && dataArrayRef.current && isActive) {
      // 실시간 오디오 레벨 표시
      analyserRef.current.getByteFrequencyData(dataArrayRef.current);

      ctx.fillStyle = barColor;
      const barWidth = width / dataArrayRef.current.length;

      for (let i = 0; i < dataArrayRef.current.length; i++) {
        const barHeight = (dataArrayRef.current[i] / 255) * height;
        const x = i * barWidth;
        
        ctx.fillRect(x, height - barHeight, barWidth - 1, barHeight);
      }
    } else if (showLevel && audioLevel > 0) {
      // 오디오 레벨로 간단한 막대 그래프 표시
      ctx.fillStyle = barColor;
      const barWidth = audioLevel * width;
      ctx.fillRect(0, height * 0.7, barWidth, height * 0.1);
    }

    // 애니메이션 반복
    if (isActive || waveformData.length > 0) {
      animationFrameRef.current = requestAnimationFrame(drawWaveform);
    }
  }, [width, height, backgroundColor, barColor, waveformData, audioLevel, showLevel, isActive]);

  // 파형 애니메이션 시작/중지
  useEffect(() => {
    if (isActive || waveformData.length > 0) {
      animationFrameRef.current = requestAnimationFrame(drawWaveform);
    } else {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      // 캔버스 초기화
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.fillStyle = backgroundColor;
          ctx.fillRect(0, 0, width, height);
        }
      }
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isActive, waveformData.length, drawWaveform, width, height, backgroundColor]);

  return (
    <div className="flex flex-col items-center gap-0.5">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className={clsx(
          'rounded',
          'block',
          isActive && 'ring-1 ring-blue-500/50'
        )}
        style={{ width: `${width}px`, height: `${height}px`, display: 'block' }}
      />
      {showLevel && (
        <div className="text-xs text-gray-400">
          {isActive ? '음성 출력 중' : '대기 중'}
        </div>
      )}
    </div>
  );
}

