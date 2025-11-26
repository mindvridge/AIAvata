/**
 * Type definitions for the Realtime AI Avatar frontend
 */

// Connection states
export type ConnectionState =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'error';

// Pipeline states (matches backend)
export type PipelineState =
  | 'idle'
  | 'listening'
  | 'processing'
  | 'speaking'
  | 'error';

// Emotion types
export type Emotion =
  | 'neutral'
  | 'happy'
  | 'sad'
  | 'angry'
  | 'surprised'
  | 'listening'
  | 'thinking'
  | 'sympathetic'
  | 'concerned';

// Session information
export interface AvatarSession {
  sessionId: string;
  avatarId: string;
  currentEmotion: Emotion;
  pipelineState: PipelineState;
  totalInteractions: number;
  createdAt: string;
  lastActivity: string;
}

// API responses
export interface HealthResponse {
  status: string;
  pipelineReady: boolean;
  version: string;
  uptimeSeconds: number;
  gpuAvailable: boolean;
}

export interface TokenResponse {
  token: string;
  roomName: string;
  livekitUrl: string;
}

export interface SessionCreateResponse {
  sessionId: string;
  avatarId: string;
  websocketUrl: string;
  livekitToken?: string;
}

// WebSocket message types
export interface WebSocketMessage {
  type: string;
  payload?: unknown;
  timestamp?: number;
}

export interface StatusMessage extends WebSocketMessage {
  type: 'status';
  status: PipelineState;
}

export interface ErrorMessage extends WebSocketMessage {
  type: 'error';
  error: string;
}

export interface EmotionChangedMessage extends WebSocketMessage {
  type: 'emotion_changed';
  emotion: Emotion;
}

// Audio configuration
export interface AudioConfig {
  sampleRate: number;
  channelCount: number;
  echoCancellation: boolean;
  noiseSuppression: boolean;
  autoGainControl: boolean;
}

// Avatar configuration
export interface AvatarConfig {
  avatarId: string;
  systemPrompt?: string;
  voiceId?: string;
  language?: 'ko' | 'en' | 'zh' | 'ja';
}

// Component props
export interface AvatarViewProps {
  videoRef?: React.RefObject<HTMLVideoElement>;
  canvasRef?: React.RefObject<HTMLCanvasElement>;
  emotion?: Emotion;
  isLoading?: boolean;
  showOverlay?: boolean;
}

export interface AudioRecorderProps {
  onAudioData: (data: ArrayBuffer) => void;
  isRecording: boolean;
  onRecordingChange: (recording: boolean) => void;
  disabled?: boolean;
}

export interface ControlsProps {
  isConnected: boolean;
  isRecording: boolean;
  pipelineState: PipelineState;
  onConnect: () => void;
  onDisconnect: () => void;
  onRecordingToggle: () => void;
  onEmotionChange?: (emotion: Emotion) => void;
}

// Hook return types
export interface UseLiveKitReturn {
  isConnected: boolean;
  connectionState: ConnectionState;
  connect: (token: string, url: string) => Promise<void>;
  disconnect: () => void;
  localAudioTrack: MediaStreamTrack | null;
  remoteVideoTrack: MediaStreamTrack | null;
  error: Error | null;
}

export interface UseAvatarSessionReturn {
  session: AvatarSession | null;
  pipelineState: PipelineState;
  emotion: Emotion;
  isConnected: boolean;
  error: string | null;
  createSession: (config?: AvatarConfig) => Promise<void>;
  sendAudio: (data: ArrayBuffer) => void;
  setEmotion: (emotion: Emotion) => void;
  disconnect: () => void;
}

export interface UseAudioRecorderReturn {
  isRecording: boolean;
  isSupported: boolean;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  audioLevel: number;
  error: Error | null;
}
