/**
 * useErrorLogger Hook
 * 
 * 에러 로그를 수집하고 관리하는 훅
 */

import { useState, useCallback, useEffect } from 'react';
import type { ErrorLogEntry } from '../components/ErrorLogPanel';

export interface UseErrorLoggerReturn {
  logs: ErrorLogEntry[];
  addError: (message: string, source?: string, details?: Record<string, unknown>) => void;
  addWarning: (message: string, source?: string, details?: Record<string, unknown>) => void;
  addInfo: (message: string, source?: string, details?: Record<string, unknown>) => void;
  clearLogs: () => void;
  errorCount: number;
  warningCount: number;
}

export function useErrorLogger(maxLogs: number = 1000): UseErrorLoggerReturn {
  const [logs, setLogs] = useState<ErrorLogEntry[]>([]);

  const addLog = useCallback(
    (
      level: ErrorLogEntry['level'],
      message: string,
      source?: string,
      details?: Record<string, unknown>,
      error?: Error
    ) => {
      const entry: ErrorLogEntry = {
        id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        timestamp: new Date(),
        level,
        message,
        source,
        details,
        stack: error?.stack,
      };

      setLogs((prev) => {
        const newLogs = [entry, ...prev];
        // Keep only the most recent logs
        return newLogs.slice(0, maxLogs);
      });
    },
    [maxLogs]
  );

  const addError = useCallback(
    (message: string, source?: string, details?: Record<string, unknown>) => {
      addLog('error', message, source, details);
      // Also log to console
      console.error(`[${source || 'App'}] ${message}`, details);
    },
    [addLog]
  );

  const addWarning = useCallback(
    (message: string, source?: string, details?: Record<string, unknown>) => {
      addLog('warning', message, source, details);
      console.warn(`[${source || 'App'}] ${message}`, details);
    },
    [addLog]
  );

  const addInfo = useCallback(
    (message: string, source?: string, details?: Record<string, unknown>) => {
      addLog('info', message, source, details);
      console.info(`[${source || 'App'}] ${message}`, details);
    },
    [addLog]
  );

  const clearLogs = useCallback(() => {
    setLogs([]);
  }, []);

  // Catch unhandled errors
  useEffect(() => {
    const handleError = (event: ErrorEvent) => {
      addError(
        event.message || 'Unknown error',
        event.filename || 'Unknown',
        {
          lineno: event.lineno,
          colno: event.colno,
          error: event.error?.toString(),
        }
      );
    };

    const handleRejection = (event: PromiseRejectionEvent) => {
      addError(
        `Unhandled promise rejection: ${event.reason}`,
        'Promise',
        {
          reason: event.reason?.toString(),
        }
      );
    };

    window.addEventListener('error', handleError);
    window.addEventListener('unhandledrejection', handleRejection);

    return () => {
      window.removeEventListener('error', handleError);
      window.removeEventListener('unhandledrejection', handleRejection);
    };
  }, [addError]);

  const errorCount = logs.filter((l) => l.level === 'error').length;
  const warningCount = logs.filter((l) => l.level === 'warning').length;

  return {
    logs,
    addError,
    addWarning,
    addInfo,
    clearLogs,
    errorCount,
    warningCount,
  };
}

export default useErrorLogger;

