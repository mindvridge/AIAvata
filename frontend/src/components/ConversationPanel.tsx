/**
 * ConversationPanel Component
 *
 * Displays conversation history and transcript
 */

import React, { useRef, useEffect } from 'react';
import { User, Bot, Volume2 } from 'lucide-react';
import { clsx } from 'clsx';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  emotion?: string;
}

interface ConversationPanelProps {
  messages: Message[];
  isLoading?: boolean;
  currentTranscript?: string;
}

export function ConversationPanel({
  messages,
  isLoading = false,
  currentTranscript = '',
}: ConversationPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, currentTranscript]);

  return (
    <div className="flex flex-col h-full bg-gray-800/30 rounded-lg min-h-0">
      {/* Header */}
      <div className="flex items-center justify-between px-3 sm:px-4 py-2 sm:py-3 border-b border-gray-700 flex-shrink-0">
        <h3 className="text-xs sm:text-sm font-medium text-white">대화 기록</h3>
        <span className="text-xs text-gray-500">{messages.length}개</span>
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-3 sm:p-4 space-y-3 sm:space-y-4 min-h-0"
      >
        {messages.length === 0 && !currentTranscript && (
          <div className="flex items-center justify-center h-full">
            <p className="text-gray-500 text-sm">대화를 시작해보세요</p>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={clsx(
              'flex gap-3',
              message.role === 'user' ? 'flex-row-reverse' : 'flex-row'
            )}
          >
            {/* Avatar */}
            <div
              className={clsx(
                'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center',
                message.role === 'user'
                  ? 'bg-primary-500'
                  : 'bg-gray-600'
              )}
            >
              {message.role === 'user' ? (
                <User className="w-4 h-4 text-white" />
              ) : (
                <Bot className="w-4 h-4 text-white" />
              )}
            </div>

            {/* Content */}
            <div
              className={clsx(
                'flex flex-col max-w-[75%]',
                message.role === 'user' ? 'items-end' : 'items-start'
              )}
            >
              <div
                className={clsx(
                  'px-4 py-2 rounded-2xl',
                  message.role === 'user'
                    ? 'bg-primary-500 text-white rounded-br-md'
                    : 'bg-gray-700 text-gray-100 rounded-bl-md'
                )}
              >
                <p className="text-sm whitespace-pre-wrap">{message.content}</p>
              </div>

              {/* Metadata */}
              <div className="flex items-center gap-2 mt-1 px-1">
                <span className="text-xs text-gray-500">
                  {message.timestamp.toLocaleTimeString('ko-KR', {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </span>
                {message.emotion && (
                  <span className="text-xs text-gray-500">
                    • {message.emotion}
                  </span>
                )}
              </div>
            </div>
          </div>
        ))}

        {/* Current transcript (real-time) */}
        {currentTranscript && (
          <div className="flex gap-3 flex-row-reverse">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary-500 flex items-center justify-center">
              <User className="w-4 h-4 text-white" />
            </div>
            <div className="flex flex-col items-end max-w-[75%]">
              <div className="px-4 py-2 rounded-2xl bg-primary-500/50 text-white rounded-br-md border border-primary-400/30">
                <p className="text-sm italic">{currentTranscript}</p>
              </div>
            </div>
          </div>
        )}

        {/* Loading indicator */}
        {isLoading && (
          <div className="flex gap-3">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center">
              <Bot className="w-4 h-4 text-white" />
            </div>
            <div className="flex items-center gap-2 px-4 py-2 rounded-2xl bg-gray-700 rounded-bl-md">
              <Volume2 className="w-4 h-4 text-gray-400 animate-pulse" />
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100" />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200" />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ConversationPanel;
