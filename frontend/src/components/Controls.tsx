/**
 * Controls Component
 *
 * Main control panel for the avatar interface
 */

import React from 'react';
import {
  Play,
  Square,
  Settings,
  Volume2,
  VolumeX,
  Maximize2,
  RefreshCw,
} from 'lucide-react';
import { clsx } from 'clsx';
import type { ConnectionState, PipelineState, Emotion } from '../types';

interface ControlsProps {
  connectionState: ConnectionState;
  pipelineState: PipelineState;
  emotion: Emotion;
  isMuted?: boolean;
  onConnect: () => void;
  onDisconnect: () => void;
  onMuteToggle?: () => void;
  onFullscreen?: () => void;
  onSettingsOpen?: () => void;
  onEmotionSelect?: (emotion: Emotion) => void;
  showEmotionSelector?: boolean;
}

const EMOTIONS: Emotion[] = [
  'neutral',
  'happy',
  'sad',
  'listening',
  'thinking',
];

const EMOTION_EMOJIS: Record<Emotion, string> = {
  neutral: '😐',
  happy: '😊',
  sad: '😢',
  angry: '😠',
  surprised: '😲',
  listening: '👂',
  thinking: '🤔',
  sympathetic: '🥺',
  concerned: '😟',
};

export function Controls({
  connectionState,
  pipelineState,
  emotion,
  isMuted = false,
  onConnect,
  onDisconnect,
  onMuteToggle,
  onFullscreen,
  onSettingsOpen,
  onEmotionSelect,
  showEmotionSelector = false,
}: ControlsProps) {
  const isConnected = connectionState === 'connected';
  const isConnecting = connectionState === 'connecting' || connectionState === 'reconnecting';
  const isProcessing = pipelineState === 'processing' || pipelineState === 'speaking';

  return (
    <div className="flex flex-col gap-4">
      {/* Main controls row */}
      <div className="flex items-center justify-center gap-4">
        {/* Connect/Disconnect button */}
        <button
          onClick={isConnected ? onDisconnect : onConnect}
          disabled={isConnecting}
          className={clsx(
            'flex items-center gap-2 px-6 py-3 rounded-full font-medium',
            'transition-all duration-200',
            'focus:outline-none focus:ring-4',
            isConnected
              ? 'bg-red-500 hover:bg-red-600 text-white focus:ring-red-500/50'
              : 'bg-primary-500 hover:bg-primary-600 text-white focus:ring-primary-500/50',
            isConnecting && 'opacity-70 cursor-wait'
          )}
        >
          {isConnecting ? (
            <>
              <RefreshCw className="w-5 h-5 animate-spin" />
              연결 중...
            </>
          ) : isConnected ? (
            <>
              <Square className="w-5 h-5" />
              종료
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              시작
            </>
          )}
        </button>

        {/* Mute button */}
        {onMuteToggle && (
          <button
            onClick={onMuteToggle}
            className={clsx(
              'p-3 rounded-full transition-colors',
              'focus:outline-none focus:ring-2 focus:ring-primary-500/50',
              isMuted
                ? 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                : 'bg-gray-700 text-white hover:bg-gray-600'
            )}
            title={isMuted ? '음소거 해제' : '음소거'}
          >
            {isMuted ? (
              <VolumeX className="w-5 h-5" />
            ) : (
              <Volume2 className="w-5 h-5" />
            )}
          </button>
        )}

        {/* Fullscreen button */}
        {onFullscreen && (
          <button
            onClick={onFullscreen}
            className={clsx(
              'p-3 rounded-full bg-gray-700 text-white',
              'hover:bg-gray-600 transition-colors',
              'focus:outline-none focus:ring-2 focus:ring-primary-500/50'
            )}
            title="전체화면"
          >
            <Maximize2 className="w-5 h-5" />
          </button>
        )}

        {/* Settings button */}
        {onSettingsOpen && (
          <button
            onClick={onSettingsOpen}
            className={clsx(
              'p-3 rounded-full bg-gray-700 text-white',
              'hover:bg-gray-600 transition-colors',
              'focus:outline-none focus:ring-2 focus:ring-primary-500/50'
            )}
            title="설정"
          >
            <Settings className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Connection status */}
      <div className="flex items-center justify-center gap-2">
        <div
          className={clsx(
            'status-dot',
            isConnected ? 'connected' : isConnecting ? 'connecting' : 'disconnected'
          )}
        />
        <span className="text-sm text-gray-400">
          {isConnected
            ? '연결됨'
            : isConnecting
            ? '연결 중...'
            : '연결 안됨'}
        </span>
      </div>

      {/* Emotion selector (optional) */}
      {showEmotionSelector && isConnected && (
        <div className="flex items-center justify-center gap-2 flex-wrap">
          <span className="text-xs text-gray-500 mr-2">감정 선택:</span>
          {EMOTIONS.map((em) => (
            <button
              key={em}
              onClick={() => onEmotionSelect?.(em)}
              disabled={isProcessing}
              className={clsx(
                'px-3 py-1.5 rounded-full text-sm transition-colors',
                'focus:outline-none focus:ring-2 focus:ring-primary-500/50',
                emotion === em
                  ? 'bg-primary-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600',
                isProcessing && 'opacity-50 cursor-not-allowed'
              )}
            >
              <span className="mr-1">{EMOTION_EMOJIS[em]}</span>
              {em}
            </button>
          ))}
        </div>
      )}

      {/* Pipeline state indicator */}
      {isConnected && pipelineState !== 'idle' && (
        <div className="flex items-center justify-center">
          <div
            className={clsx(
              'px-4 py-2 rounded-full text-sm font-medium',
              'flex items-center gap-2',
              pipelineState === 'listening' && 'bg-blue-500/20 text-blue-400',
              pipelineState === 'processing' && 'bg-yellow-500/20 text-yellow-400',
              pipelineState === 'speaking' && 'bg-green-500/20 text-green-400',
              pipelineState === 'error' && 'bg-red-500/20 text-red-400'
            )}
          >
            {pipelineState === 'listening' && (
              <>
                <span className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                듣고 있습니다...
              </>
            )}
            {pipelineState === 'processing' && (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                처리 중...
              </>
            )}
            {pipelineState === 'speaking' && (
              <>
                <Volume2 className="w-4 h-4" />
                응답 중...
              </>
            )}
            {pipelineState === 'error' && (
              <>
                <span className="w-2 h-2 bg-red-400 rounded-full" />
                오류 발생
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default Controls;
