/**
 * ChatInput Component
 *
 * 텍스트 채팅 입력 컴포넌트
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { clsx } from 'clsx';

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading?: boolean;
  isDisabled?: boolean;
  placeholder?: string;
  maxLength?: number;
}

export function ChatInput({
  onSend,
  isLoading = false,
  isDisabled = false,
  placeholder = '메시지를 입력하세요...',
  maxLength = 500,
}: ChatInputProps) {
  const [message, setMessage] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  // 전송 핸들러
  const handleSend = useCallback(() => {
    const trimmedMessage = message.trim();
    if (trimmedMessage && !isLoading && !isDisabled) {
      onSend(trimmedMessage);
      setMessage('');
    }
  }, [message, onSend, isLoading, isDisabled]);

  // Enter 키 핸들러
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  // 입력 변경 핸들러
  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    if (value.length <= maxLength) {
      setMessage(value);
    }
  }, [maxLength]);

  // 포커스 관리
  useEffect(() => {
    if (!isDisabled && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isDisabled]);

  const canSend = message.trim().length > 0 && !isLoading && !isDisabled;

  return (
    <div className="flex items-center gap-1.5 sm:gap-2 p-2 sm:p-2.5 bg-gray-800/50 rounded-lg border border-gray-700">
      {/* 입력 필드 */}
      <input
        ref={inputRef}
        type="text"
        value={message}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder={isDisabled ? '연결 후 채팅 가능' : placeholder}
        disabled={isDisabled || isLoading}
        className={clsx(
          'flex-1 bg-transparent border-none outline-none text-white placeholder-gray-500',
          'text-xs sm:text-sm py-1 sm:py-1.5 px-2',
          isDisabled && 'cursor-not-allowed opacity-50'
        )}
        maxLength={maxLength}
      />

      {/* 글자 수 표시 - 작은 화면에서는 숨김 */}
      {message.length > 0 && (
        <span className={clsx(
          'text-xs hidden sm:inline',
          message.length > maxLength * 0.9 ? 'text-yellow-400' : 'text-gray-500'
        )}>
          {message.length}/{maxLength}
        </span>
      )}

      {/* 전송 버튼 */}
      <button
        onClick={handleSend}
        disabled={!canSend}
        className={clsx(
          'p-1.5 sm:p-2 rounded-lg transition-all duration-200 flex-shrink-0',
          canSend
            ? 'bg-blue-600 hover:bg-blue-500 text-white cursor-pointer'
            : 'bg-gray-700 text-gray-500 cursor-not-allowed'
        )}
        title={canSend ? '전송 (Enter)' : '메시지를 입력하세요'}
      >
        {isLoading ? (
          <Loader2 className="w-4 h-4 sm:w-5 sm:h-5 animate-spin" />
        ) : (
          <Send className="w-4 h-4 sm:w-5 sm:h-5" />
        )}
      </button>
    </div>
  );
}

export default ChatInput;

