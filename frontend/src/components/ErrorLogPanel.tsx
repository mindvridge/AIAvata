/**
 * ErrorLogPanel Component
 * 
 * 에러 로그를 표시하는 패널 컴포넌트
 */

import React, { useState, useEffect, useRef } from 'react';
import { X, AlertCircle, Trash2, Download, Copy, ChevronDown, ChevronUp } from 'lucide-react';
import { clsx } from 'clsx';

export interface ErrorLogEntry {
  id: string;
  timestamp: Date;
  level: 'error' | 'warning' | 'info';
  message: string;
  source?: string;
  stack?: string;
  details?: Record<string, unknown>;
}

interface ErrorLogPanelProps {
  isOpen: boolean;
  onClose: () => void;
  logs: ErrorLogEntry[];
  onClear: () => void;
}

export function ErrorLogPanel({
  isOpen,
  onClose,
  logs,
  onClear,
}: ErrorLogPanelProps) {
  const [isMinimized, setIsMinimized] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [filter, setFilter] = useState<'all' | 'error' | 'warning' | 'info'>('all');
  const logEndRef = useRef<HTMLDivElement>(null);

  // Auto scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  // Filter logs based on selected level
  const filteredLogs = logs.filter((log) => {
    if (filter === 'all') return true;
    return log.level === filter;
  });

  // Get log count by level
  const errorCount = logs.filter((l) => l.level === 'error').length;
  const warningCount = logs.filter((l) => l.level === 'warning').length;
  const infoCount = logs.filter((l) => l.level === 'info').length;

  // Format timestamp
  const formatTime = (date: Date): string => {
    return new Intl.DateTimeFormat('ko-KR', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      fractionalSecondDigits: 3,
    }).format(date);
  };

  // Copy log to clipboard
  const copyLog = (log: ErrorLogEntry) => {
    const text = `[${formatTime(log.timestamp)}] ${log.level.toUpperCase()}: ${log.message}${
      log.stack ? `\n${log.stack}` : ''
    }${log.details ? `\n${JSON.stringify(log.details, null, 2)}` : ''}`;
    navigator.clipboard.writeText(text);
  };

  // Download all logs as JSON
  const downloadLogs = () => {
    const dataStr = JSON.stringify(logs, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `error-logs-${new Date().toISOString()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Get level color
  const getLevelColor = (level: ErrorLogEntry['level']): string => {
    switch (level) {
      case 'error':
        return 'text-red-400 bg-red-500/10 border-red-500/20';
      case 'warning':
        return 'text-yellow-400 bg-yellow-500/10 border-yellow-500/20';
      case 'info':
        return 'text-blue-400 bg-blue-500/10 border-blue-500/20';
      default:
        return 'text-gray-400 bg-gray-500/10 border-gray-500/20';
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className={clsx(
        'fixed bottom-0 right-0 w-full md:w-[600px] bg-gray-900 border-t border-l border-gray-700 shadow-2xl transition-all duration-300 z-50',
        isMinimized ? 'h-12' : 'h-[500px]'
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-400" />
          <div>
            <h3 className="text-sm font-semibold text-white">에러 로그</h3>
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <span className={clsx(errorCount > 0 && 'text-red-400')}>
                에러: {errorCount}
              </span>
              <span className={clsx(warningCount > 0 && 'text-yellow-400')}>
                경고: {warningCount}
              </span>
              <span>정보: {infoCount}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Filter buttons */}
          <div className="flex items-center gap-1 bg-gray-700 rounded-lg p-1">
            {(['all', 'error', 'warning', 'info'] as const).map((level) => (
              <button
                key={level}
                onClick={() => setFilter(level)}
                className={clsx(
                  'px-2 py-1 text-xs rounded transition-colors',
                  filter === level
                    ? 'bg-primary-500 text-white'
                    : 'text-gray-400 hover:text-white'
                )}
              >
                {level === 'all' ? '전체' : level === 'error' ? '에러' : level === 'warning' ? '경고' : '정보'}
              </button>
            ))}
          </div>

          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={clsx(
              'px-2 py-1 text-xs rounded transition-colors',
              autoScroll
                ? 'bg-green-500/20 text-green-400'
                : 'bg-gray-700 text-gray-400 hover:text-white'
            )}
            title="자동 스크롤"
          >
            {autoScroll ? '자동' : '수동'}
          </button>

          <button
            onClick={downloadLogs}
            className="p-1.5 text-gray-400 hover:text-white transition-colors"
            title="로그 다운로드"
          >
            <Download className="w-4 h-4" />
          </button>

          <button
            onClick={onClear}
            className="p-1.5 text-gray-400 hover:text-red-400 transition-colors"
            title="로그 지우기"
          >
            <Trash2 className="w-4 h-4" />
          </button>

          <button
            onClick={() => setIsMinimized(!isMinimized)}
            className="p-1.5 text-gray-400 hover:text-white transition-colors"
            title={isMinimized ? '확장' : '최소화'}
          >
            {isMinimized ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </button>

          <button
            onClick={onClose}
            className="p-1.5 text-gray-400 hover:text-white transition-colors"
            title="닫기"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Log content */}
      {!isMinimized && (
        <div className="h-[calc(100%-3.5rem)] overflow-y-auto">
          {filteredLogs.length === 0 ? (
            <div className="flex items-center justify-center h-full text-gray-500">
              <div className="text-center">
                <AlertCircle className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>로그가 없습니다</p>
              </div>
            </div>
          ) : (
            <div className="p-2 space-y-2">
              {filteredLogs.map((log) => (
                <div
                  key={log.id}
                  className={clsx(
                    'p-3 rounded-lg border text-sm',
                    getLevelColor(log.level)
                  )}
                >
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <span className="text-xs font-mono opacity-70">
                        {formatTime(log.timestamp)}
                      </span>
                      <span className="text-xs font-semibold uppercase">
                        {log.level}
                      </span>
                      {log.source && (
                        <span className="text-xs opacity-60 truncate">
                          {log.source}
                        </span>
                      )}
                    </div>
                    <button
                      onClick={() => copyLog(log)}
                      className="p-1 hover:bg-white/10 rounded transition-colors flex-shrink-0"
                      title="복사"
                    >
                      <Copy className="w-3 h-3" />
                    </button>
                  </div>

                  <div className="text-sm font-medium mb-1 break-words">
                    {log.message}
                  </div>

                  {log.stack && (
                    <details className="mt-2">
                      <summary className="text-xs opacity-70 cursor-pointer hover:opacity-100">
                        스택 트레이스 보기
                      </summary>
                      <pre className="mt-2 p-2 bg-black/30 rounded text-xs font-mono overflow-x-auto">
                        {log.stack}
                      </pre>
                    </details>
                  )}

                  {log.details && Object.keys(log.details).length > 0 && (
                    <details className="mt-2">
                      <summary className="text-xs opacity-70 cursor-pointer hover:opacity-100">
                        상세 정보 보기
                      </summary>
                      <pre className="mt-2 p-2 bg-black/30 rounded text-xs font-mono overflow-x-auto">
                        {JSON.stringify(log.details, null, 2)}
                      </pre>
                    </details>
                  )}
                </div>
              ))}
              <div ref={logEndRef} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ErrorLogPanel;

