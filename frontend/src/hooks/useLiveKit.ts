/**
 * LiveKit connection hook
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import {
  Room,
  RoomEvent,
  Track,
  RemoteTrack,
  RemoteTrackPublication,
  LocalTrack,
  ConnectionState as LKConnectionState,
} from 'livekit-client';
import type { ConnectionState, UseLiveKitReturn } from '../types';

export function useLiveKit(): UseLiveKitReturn {
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [localAudioTrack, setLocalAudioTrack] = useState<MediaStreamTrack | null>(null);
  const [remoteVideoTrack, setRemoteVideoTrack] = useState<MediaStreamTrack | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const roomRef = useRef<Room | null>(null);

  const mapConnectionState = (state: LKConnectionState): ConnectionState => {
    switch (state) {
      case LKConnectionState.Connected:
        return 'connected';
      case LKConnectionState.Connecting:
        return 'connecting';
      case LKConnectionState.Reconnecting:
        return 'reconnecting';
      case LKConnectionState.Disconnected:
      default:
        return 'disconnected';
    }
  };

  const handleTrackSubscribed = useCallback(
    (
      track: RemoteTrack,
      publication: RemoteTrackPublication
    ) => {
      if (track.kind === Track.Kind.Video) {
        const mediaTrack = track.mediaStreamTrack;
        setRemoteVideoTrack(mediaTrack);
        console.log('Remote video track subscribed');
      }
    },
    []
  );

  const handleTrackUnsubscribed = useCallback(
    (track: RemoteTrack) => {
      if (track.kind === Track.Kind.Video) {
        setRemoteVideoTrack(null);
        console.log('Remote video track unsubscribed');
      }
    },
    []
  );

  const handleDisconnected = useCallback(() => {
    setConnectionState('disconnected');
    setLocalAudioTrack(null);
    setRemoteVideoTrack(null);
    console.log('Disconnected from room');
  }, []);

  const connect = useCallback(async (token: string, url: string) => {
    try {
      setError(null);
      setConnectionState('connecting');

      // Create room if not exists
      if (!roomRef.current) {
        roomRef.current = new Room({
          adaptiveStream: true,
          dynacast: true,
          videoCaptureDefaults: {
            resolution: { width: 640, height: 480, frameRate: 30 },
          },
          audioCaptureDefaults: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
      }

      const room = roomRef.current;

      // Set up event listeners
      room.on(RoomEvent.TrackSubscribed, handleTrackSubscribed);
      room.on(RoomEvent.TrackUnsubscribed, handleTrackUnsubscribed);
      room.on(RoomEvent.Disconnected, handleDisconnected);
      room.on(RoomEvent.ConnectionStateChanged, (state) => {
        setConnectionState(mapConnectionState(state));
      });

      // Connect to room
      await room.connect(url, token);
      setConnectionState('connected');

      // Enable local microphone
      await room.localParticipant.setMicrophoneEnabled(true);

      const audioTrack = room.localParticipant.audioTrackPublications.values().next().value;
      if (audioTrack?.track) {
        setLocalAudioTrack(audioTrack.track.mediaStreamTrack);
      }

      console.log('Connected to LiveKit room');
    } catch (err) {
      console.error('Failed to connect to LiveKit:', err);
      setError(err instanceof Error ? err : new Error('Connection failed'));
      setConnectionState('error');
    }
  }, [handleTrackSubscribed, handleTrackUnsubscribed, handleDisconnected]);

  const disconnect = useCallback(() => {
    if (roomRef.current) {
      roomRef.current.disconnect();
      roomRef.current = null;
    }
    setConnectionState('disconnected');
    setLocalAudioTrack(null);
    setRemoteVideoTrack(null);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    isConnected: connectionState === 'connected',
    connectionState,
    connect,
    disconnect,
    localAudioTrack,
    remoteVideoTrack,
    error,
  };
}

export default useLiveKit;
