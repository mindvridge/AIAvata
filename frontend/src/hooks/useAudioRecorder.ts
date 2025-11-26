/**
 * Audio recorder hook with VAD support
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import type { UseAudioRecorderReturn, AudioConfig } from '../types';

const DEFAULT_CONFIG: AudioConfig = {
  sampleRate: 16000,
  channelCount: 1,
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
};

interface UseAudioRecorderOptions {
  config?: Partial<AudioConfig>;
  onAudioData?: (data: ArrayBuffer) => void;
  chunkInterval?: number; // ms between audio chunks
  silenceThreshold?: number; // 0-1, audio level below this is silence
  silenceTimeout?: number; // ms of silence before stopping
}

export function useAudioRecorder(
  options: UseAudioRecorderOptions = {}
): UseAudioRecorderReturn {
  const {
    config = {},
    onAudioData,
    chunkInterval = 100,
    silenceThreshold = 0.01,
    silenceTimeout = 1500,
  } = options;

  const [isRecording, setIsRecording] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [error, setError] = useState<Error | null>(null);

  const audioConfig = { ...DEFAULT_CONFIG, ...config };

  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const chunkIntervalRef = useRef<number | null>(null);
  const silenceTimeoutRef = useRef<number | null>(null);
  const audioBufferRef = useRef<Float32Array[]>([]);
  const lastSoundTimeRef = useRef<number>(Date.now());

  const isSupported = typeof navigator !== 'undefined' &&
    'mediaDevices' in navigator &&
    'getUserMedia' in navigator.mediaDevices;

  const calculateAudioLevel = useCallback((analyser: AnalyserNode): number => {
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(dataArray);

    const sum = dataArray.reduce((acc, val) => acc + val, 0);
    const average = sum / dataArray.length;
    return average / 255; // Normalize to 0-1
  }, []);

  const processAudioBuffer = useCallback(() => {
    if (audioBufferRef.current.length === 0) return;

    // Combine all buffered chunks
    const totalLength = audioBufferRef.current.reduce(
      (acc, chunk) => acc + chunk.length,
      0
    );
    const combined = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of audioBufferRef.current) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }
    audioBufferRef.current = [];

    // Convert to 16-bit PCM
    const pcmData = new Int16Array(combined.length);
    for (let i = 0; i < combined.length; i++) {
      const sample = Math.max(-1, Math.min(1, combined[i]));
      pcmData[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }

    onAudioData?.(pcmData.buffer);
  }, [onAudioData]);

  const startRecording = useCallback(async () => {
    if (!isSupported) {
      setError(new Error('Audio recording not supported'));
      return;
    }

    try {
      setError(null);

      // Get microphone stream
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: audioConfig.sampleRate,
          channelCount: audioConfig.channelCount,
          echoCancellation: audioConfig.echoCancellation,
          noiseSuppression: audioConfig.noiseSuppression,
          autoGainControl: audioConfig.autoGainControl,
        },
      });

      streamRef.current = stream;

      // Create audio context
      const audioContext = new AudioContext({
        sampleRate: audioConfig.sampleRate,
      });
      audioContextRef.current = audioContext;

      // Create source and analyser
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.8;
      analyserRef.current = analyser;

      // Create processor for audio data
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (event) => {
        const inputData = event.inputBuffer.getChannelData(0);
        audioBufferRef.current.push(new Float32Array(inputData));
      };

      // Connect nodes
      source.connect(analyser);
      analyser.connect(processor);
      processor.connect(audioContext.destination);

      // Start chunk interval
      chunkIntervalRef.current = window.setInterval(() => {
        if (analyserRef.current) {
          const level = calculateAudioLevel(analyserRef.current);
          setAudioLevel(level);

          // Check for silence
          if (level > silenceThreshold) {
            lastSoundTimeRef.current = Date.now();
          }

          // Process audio buffer
          processAudioBuffer();
        }
      }, chunkInterval);

      // Start silence detection
      silenceTimeoutRef.current = window.setInterval(() => {
        const silenceDuration = Date.now() - lastSoundTimeRef.current;
        if (silenceDuration > silenceTimeout && isRecording) {
          // Optionally auto-stop on silence
          // stopRecording();
        }
      }, 500);

      setIsRecording(true);
      lastSoundTimeRef.current = Date.now();
      console.log('Recording started');

    } catch (err) {
      console.error('Failed to start recording:', err);
      setError(err instanceof Error ? err : new Error('Failed to start recording'));
    }
  }, [
    isSupported,
    audioConfig,
    chunkInterval,
    silenceThreshold,
    silenceTimeout,
    calculateAudioLevel,
    processAudioBuffer,
    isRecording,
  ]);

  const stopRecording = useCallback(() => {
    // Clear intervals
    if (chunkIntervalRef.current) {
      clearInterval(chunkIntervalRef.current);
      chunkIntervalRef.current = null;
    }
    if (silenceTimeoutRef.current) {
      clearInterval(silenceTimeoutRef.current);
      silenceTimeoutRef.current = null;
    }

    // Process remaining buffer
    processAudioBuffer();

    // Disconnect processor
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    // Close audio context
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    // Stop stream tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    analyserRef.current = null;
    audioBufferRef.current = [];
    setAudioLevel(0);
    setIsRecording(false);
    console.log('Recording stopped');
  }, [processAudioBuffer]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (isRecording) {
        stopRecording();
      }
    };
  }, [isRecording, stopRecording]);

  return {
    isRecording,
    isSupported,
    startRecording,
    stopRecording,
    audioLevel,
    error,
  };
}

export default useAudioRecorder;
