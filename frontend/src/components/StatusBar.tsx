/**
 * StatusBar Component
 *
 * Displays connection and system status
 */

import React, { useEffect, useState } from 'react';
import { Wifi, WifiOff, Cpu, HardDrive, Clock } from 'lucide-react';
import { clsx } from 'clsx';
import { api } from '../lib/api';
import type { ConnectionState } from '../types';

interface StatusBarProps {
  connectionState: ConnectionState;
  latency?: number;
  showMetrics?: boolean;
}

interface ServerHealth {
  status: string;
  pipelineReady: boolean;
  version: string;
  uptimeSeconds: number;
  gpuAvailable: boolean;
}

export function StatusBar({
  connectionState,
  latency,
  showMetrics = false,
}: StatusBarProps) {
  const [health, setHealth] = useState<ServerHealth | null>(null);
  const [metrics, setMetrics] = useState<Record<string, number> | null>(null);

  const isConnected = connectionState === 'connected';

  // Fetch server health periodically
  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const response = await api.healthCheck();
        setHealth(response);
      } catch (err) {
        setHealth(null);
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  // Fetch metrics if enabled (placeholder for future implementation)
  useEffect(() => {
    if (!showMetrics) return;
    // Metrics API not yet implemented
    setMetrics(null);
  }, [showMetrics]);

  const formatUptime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}시간 ${minutes}분`;
  };

  return (
    <div className="flex items-center justify-between px-4 py-2 bg-gray-800/50 rounded-lg">
      {/* Left: Connection status */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          {isConnected ? (
            <Wifi className="w-4 h-4 text-green-400" />
          ) : (
            <WifiOff className="w-4 h-4 text-gray-500" />
          )}
          <span
            className={clsx(
              'text-sm font-medium',
              isConnected ? 'text-green-400' : 'text-gray-500'
            )}
          >
            {connectionState === 'connected'
              ? '연결됨'
              : connectionState === 'connecting'
              ? '연결 중...'
              : connectionState === 'reconnecting'
              ? '재연결 중...'
              : '연결 안됨'}
          </span>
        </div>

        {latency !== undefined && isConnected && (
          <div className="flex items-center gap-1.5 text-sm text-gray-400">
            <Clock className="w-3.5 h-3.5" />
            <span>{latency}ms</span>
          </div>
        )}
      </div>

      {/* Center: Server status */}
      {health && (
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <div
              className={clsx(
                'w-2 h-2 rounded-full',
                health.pipelineReady ? 'bg-green-400' : 'bg-yellow-400'
              )}
            />
            <span className="text-xs text-gray-400">
              {health.pipelineReady ? '파이프라인 준비됨' : '초기화 중'}
            </span>
          </div>

          <div className="flex items-center gap-1.5">
            <Cpu
              className={clsx(
                'w-3.5 h-3.5',
                health.gpuAvailable ? 'text-green-400' : 'text-gray-500'
              )}
            />
            <span className="text-xs text-gray-400">
              {health.gpuAvailable ? 'GPU' : 'CPU'}
            </span>
          </div>

          <div className="flex items-center gap-1.5">
            <HardDrive className="w-3.5 h-3.5 text-gray-400" />
            <span className="text-xs text-gray-400">
              v{health.version}
            </span>
          </div>
        </div>
      )}

      {/* Right: Metrics */}
      {showMetrics && metrics && (
        <div className="flex items-center gap-4">
          <div className="text-xs text-gray-400">
            <span className="text-gray-500">요청: </span>
            {metrics.total_requests || 0}
          </div>
          <div className="text-xs text-gray-400">
            <span className="text-gray-500">평균 지연: </span>
            {(metrics.avg_latency_ms || 0).toFixed(0)}ms
          </div>
        </div>
      )}
    </div>
  );
}

export default StatusBar;
