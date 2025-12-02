/**
 * API client for the Realtime AI Avatar backend
 */

import type {
  SessionCreateResponse,
  HealthResponse,
  AvatarConfig
} from '../types';

// API base URL - Vite 환경변수 또는 기본값 사용
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * API 요청 헬퍼 함수
 */
async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `API error: ${response.status}`);
  }

  return response.json();
}

/**
 * API 클라이언트
 */
export const api = {
  /**
   * 서버 상태 확인
   */
  async health(): Promise<HealthResponse> {
    return request<HealthResponse>('/health');
  },

  /**
   * 아바타 세션 생성
   */
  async createSession(config?: AvatarConfig): Promise<SessionCreateResponse> {
    return request<SessionCreateResponse>('/api/avatar/create', {
      method: 'POST',
      body: JSON.stringify({
        avatar_id: config?.avatarId || 'default',
        system_prompt: config?.systemPrompt,
        voice_id: config?.voiceId,
        language: config?.language,
      }),
    });
  },

  /**
   * 세션 정보 조회
   */
  async getSession(sessionId: string): Promise<any> {
    return request(`/api/avatar/${sessionId}`);
  },

  /**
   * 세션 삭제
   */
  async deleteSession(sessionId: string): Promise<void> {
    await request(`/api/avatar/${sessionId}`, {
      method: 'DELETE',
    });
  },

  /**
   * 사용 가능한 아바타 목록
   */
  async listAvatars(): Promise<{ avatars: any[] }> {
    return request('/api/avatars');
  },

  /**
   * 지원되는 감정 목록
   */
  async listEmotions(): Promise<{ emotions: string[] }> {
    return request('/api/emotions');
  },

  /**
   * 메트릭 조회
   */
  async getMetrics(): Promise<any> {
    return request('/api/metrics');
  },
};

export default api;
