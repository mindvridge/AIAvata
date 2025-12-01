/**
 * AudioRecorder Component
 *
 * Microphone input with visual feedback
 */

import React, { useEffect, useCallback } from 'react';
import { Mic, MicOff, AlertCircle } from 'lucide-react';
import { clsx } from 'clsx';
import { useAudioRecorder } from '../hooks/useAudioRecorder';

interface AudioRecorderProps {
  onAudioData: (data: ArrayBuffer) => void;
  isEnabled?: boolean;
  size?: 'sm' | 'md' | 'lg';
  showLevel?: boolean;
}

const SIZE_CLASSES = {
  sm: 'w-12 h-12',
  md: 'w-16 h-16',
  lg: 'w-20 h-20',
};

const ICON_SIZES = {
  sm: 'w-5 h-5',
  md: 'w-7 h-7',
  lg: 'w-9 h-9',
};

export function AudioRecorder({
  onAudioData,
  isEnabled = true,
  size = 'lg',
  showLevel = true,
}: AudioRecorderProps) {
  const {
    isRecording,
    isSupported,
    startRecording,
    stopRecording,
    audioLevel,
    error,
  } = useAudioRecorder({
    onAudioData,
    chunkInterval: 100,
  });

  const handleClick = useCallback(() => {
    if (!isEnabled || !isSupported) return;

    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isEnabled, isSupported, isRecording, startRecording, stopRecording]);

  // Auto-stop when disabled
  useEffect(() => {
    if (!isEnabled && isRecording) {
      stopRecording();
    }
  }, [isEnabled, isRecording, stopRecording]);

  // Calculate level indicator scale
  const levelScale = 1 + audioLevel * 0.3;
  const levelOpacity = 0.3 + audioLevel * 0.7;

  if (!isSupported) {
    return (
      <div className="flex flex-col items-center gap-2">
        <div
          className={clsx(
            SIZE_CLASSES[size],
            'rounded-full bg-gray-700 flex items-center justify-center'
          )}
        >
          <AlertCircle className={clsx(ICON_SIZES[size], 'text-gray-500')} />
        </div>
        <span className="text-xs text-gray-500">마이크 지원 안됨</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-1.5 sm:gap-2">
      {/* Main button */}
      <div className="relative">
        {/* Level indicator ring */}
        {isRecording && showLevel && (
          <div
            className="absolute inset-0 rounded-full bg-primary-500/30 transition-transform duration-75"
            style={{
              transform: `scale(${levelScale})`,
              opacity: levelOpacity,
            }}
          />
        )}

        {/* Button */}
        <button
          onClick={handleClick}
          disabled={!isEnabled}
          className={clsx(
            SIZE_CLASSES[size],
            'relative rounded-full transition-all duration-200',
            'flex items-center justify-center',
            'focus:outline-none focus:ring-4 focus:ring-primary-500/50',
            isRecording
              ? 'bg-red-500 hover:bg-red-600 mic-button recording'
              : 'bg-primary-500 hover:bg-primary-600 mic-button',
            !isEnabled && 'opacity-50 cursor-not-allowed'
          )}
        >
          {isRecording ? (
            <MicOff className={clsx(ICON_SIZES[size], 'text-white')} />
          ) : (
            <Mic className={clsx(ICON_SIZES[size], 'text-white')} />
          )}
        </button>

        {/* Recording pulse */}
        {isRecording && (
          <div className="absolute inset-0 rounded-full border-4 border-red-500 animate-ping opacity-50" />
        )}
      </div>

      {/* Status text */}
      <span className={clsx(
        'text-sm font-medium',
        isRecording ? 'text-red-400' : 'text-gray-400'
      )}>
        {isRecording ? '녹음 중...' : '눌러서 말하기'}
      </span>

      {/* Audio level bar */}
      {isRecording && showLevel && (
        <div className="w-24 h-1.5 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 rounded-full transition-all duration-75"
            style={{ width: `${audioLevel * 100}%` }}
          />
        </div>
      )}

      {/* Error message */}
      {error && (
        <div className="flex items-center gap-1 text-red-400 text-xs">
          <AlertCircle className="w-3 h-3" />
          <span>{error.message}</span>
        </div>
      )}
    </div>
  );
}

export default AudioRecorder;
