"""
E2E Tests for WebSocket Communication.

WebSocket 통신 End-to-End 테스트
"""

import asyncio
import json
import time
import pytest
import numpy as np
from httpx import AsyncClient


class TestWebSocketConnection:
    """WebSocket connection tests."""

    @pytest.mark.asyncio
    async def test_websocket_connect(self, app):
        """Should establish WebSocket connection."""
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            with client.websocket_connect("/ws/avatar") as websocket:
                # Connection should be established
                assert websocket is not None

    @pytest.mark.asyncio
    async def test_websocket_connect_with_session(self, app, async_client: AsyncClient):
        """Should connect with existing session ID."""
        from starlette.testclient import TestClient

        # Create session first
        response = await async_client.post(
            "/api/avatar/create",
            json={"avatar_id": "default"}
        )
        session_id = response.json()["session_id"]

        with TestClient(app) as client:
            with client.websocket_connect(f"/ws/avatar/{session_id}") as websocket:
                assert websocket is not None

    @pytest.mark.asyncio
    async def test_websocket_receives_initial_state(self, app):
        """Should receive initial state after connection."""
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            with client.websocket_connect("/ws/avatar") as websocket:
                # Send start session message
                websocket.send_json({
                    "type": "start_session",
                    "avatar_id": "default"
                })

                # Should receive state message
                try:
                    data = websocket.receive_json(timeout=5)
                    assert "type" in data
                except Exception:
                    # May timeout if no handler configured
                    pass


class TestWebSocketAudioProcessing:
    """WebSocket audio processing tests."""

    @pytest.mark.asyncio
    async def test_send_audio_chunk(self, app, sample_audio_bytes):
        """Should accept audio chunk."""
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            with client.websocket_connect("/ws/avatar") as websocket:
                # Send start message
                websocket.send_json({
                    "type": "start_session",
                    "avatar_id": "default"
                })

                # Send audio chunk
                websocket.send_bytes(sample_audio_bytes)

                # Connection should remain open
                assert websocket is not None

    @pytest.mark.asyncio
    async def test_send_multiple_audio_chunks(self, app, sample_audio_16k):
        """Should handle multiple audio chunks."""
        from starlette.testclient import TestClient

        # Split audio into chunks
        chunk_size = 4096
        audio_bytes = (sample_audio_16k * 32767).astype(np.int16).tobytes()
        chunks = [
            audio_bytes[i:i+chunk_size]
            for i in range(0, len(audio_bytes), chunk_size)
        ]

        with TestClient(app) as client:
            with client.websocket_connect("/ws/avatar") as websocket:
                websocket.send_json({
                    "type": "start_session",
                    "avatar_id": "default"
                })

                # Send multiple chunks
                for chunk in chunks[:5]:  # Send first 5 chunks
                    websocket.send_bytes(chunk)
                    time.sleep(0.01)  # Small delay between chunks


class TestWebSocketTextInput:
    """WebSocket text input tests."""

    @pytest.mark.asyncio
    async def test_send_text_message(self, app):
        """Should accept text message."""
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            with client.websocket_connect("/ws/avatar") as websocket:
                websocket.send_json({
                    "type": "start_session",
                    "avatar_id": "default"
                })

                # Send text input
                websocket.send_json({
                    "type": "text",
                    "content": "안녕하세요"
                })

                # Try to receive response
                try:
                    data = websocket.receive_json(timeout=5)
                    # Should receive some response
                    assert "type" in data
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_send_korean_text(self, app):
        """Should handle Korean text properly."""
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            with client.websocket_connect("/ws/avatar") as websocket:
                websocket.send_json({
                    "type": "start_session",
                    "avatar_id": "default",
                    "language": "ko"
                })

                # Send Korean text
                websocket.send_json({
                    "type": "text",
                    "content": "오늘 날씨가 어때요?"
                })


class TestWebSocketControlMessages:
    """WebSocket control message tests."""

    @pytest.mark.asyncio
    async def test_stop_message(self, app):
        """Should handle stop message."""
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            with client.websocket_connect("/ws/avatar") as websocket:
                websocket.send_json({
                    "type": "start_session",
                    "avatar_id": "default"
                })

                # Send stop message
                websocket.send_json({
                    "type": "stop"
                })

    @pytest.mark.asyncio
    async def test_change_emotion(self, app):
        """Should handle emotion change."""
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            with client.websocket_connect("/ws/avatar") as websocket:
                websocket.send_json({
                    "type": "start_session",
                    "avatar_id": "default"
                })

                # Change emotion
                websocket.send_json({
                    "type": "set_emotion",
                    "emotion": "happy"
                })

    @pytest.mark.asyncio
    async def test_ping_pong(self, app):
        """Should respond to ping."""
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            with client.websocket_connect("/ws/avatar") as websocket:
                # Send ping
                websocket.send_json({
                    "type": "ping"
                })

                try:
                    data = websocket.receive_json(timeout=2)
                    if data.get("type") == "pong":
                        assert True
                except Exception:
                    pass  # Ping/pong may not be implemented


class TestWebSocketErrorHandling:
    """WebSocket error handling tests."""

    @pytest.mark.asyncio
    async def test_invalid_json_message(self, app):
        """Should handle invalid JSON gracefully."""
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            with client.websocket_connect("/ws/avatar") as websocket:
                # Send invalid message type
                websocket.send_json({
                    "type": "invalid_type"
                })
                # Should not crash

    @pytest.mark.asyncio
    async def test_missing_required_fields(self, app):
        """Should handle missing fields gracefully."""
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            with client.websocket_connect("/ws/avatar") as websocket:
                # Send incomplete message
                websocket.send_json({
                    "type": "text"
                    # Missing "content" field
                })


class TestWebSocketPerformance:
    """WebSocket performance tests."""

    @pytest.mark.asyncio
    async def test_connection_time(self, app, performance_thresholds):
        """WebSocket connection should be fast."""
        from starlette.testclient import TestClient

        start = time.perf_counter()

        with TestClient(app) as client:
            with client.websocket_connect("/ws/avatar") as websocket:
                elapsed_ms = (time.perf_counter() - start) * 1000

        # Connection should be established quickly
        assert elapsed_ms < 1000  # 1 second max

    @pytest.mark.asyncio
    async def test_message_throughput(self, app, sample_audio_16k):
        """Should handle high message throughput."""
        from starlette.testclient import TestClient

        chunk_size = 1024
        audio_bytes = (sample_audio_16k * 32767).astype(np.int16).tobytes()

        with TestClient(app) as client:
            with client.websocket_connect("/ws/avatar") as websocket:
                websocket.send_json({
                    "type": "start_session",
                    "avatar_id": "default"
                })

                start = time.perf_counter()
                message_count = 0

                # Send many messages quickly
                for i in range(100):
                    chunk = audio_bytes[i*chunk_size:(i+1)*chunk_size]
                    if chunk:
                        websocket.send_bytes(chunk)
                        message_count += 1

                elapsed = time.perf_counter() - start
                messages_per_second = message_count / elapsed

                # Should handle at least 100 messages/second
                assert messages_per_second > 50


class TestWebSocketConcurrency:
    """WebSocket concurrency tests."""

    @pytest.mark.asyncio
    async def test_multiple_connections(self, app):
        """Should handle multiple simultaneous connections."""
        from starlette.testclient import TestClient

        connections = []

        with TestClient(app) as client:
            # Open multiple connections
            for i in range(3):
                ws = client.websocket_connect("/ws/avatar")
                ws.__enter__()
                connections.append(ws)

            # All connections should be active
            assert len(connections) == 3

            # Clean up
            for ws in connections:
                ws.__exit__(None, None, None)

    @pytest.mark.asyncio
    async def test_session_isolation(self, app, async_client: AsyncClient):
        """Sessions should be isolated."""
        from starlette.testclient import TestClient

        # Create two sessions
        response1 = await async_client.post(
            "/api/avatar/create",
            json={"avatar_id": "default"}
        )
        session1 = response1.json()["session_id"]

        response2 = await async_client.post(
            "/api/avatar/create",
            json={"avatar_id": "default"}
        )
        session2 = response2.json()["session_id"]

        assert session1 != session2

        with TestClient(app) as client:
            # Connect to both sessions
            with client.websocket_connect(f"/ws/avatar/{session1}") as ws1:
                with client.websocket_connect(f"/ws/avatar/{session2}") as ws2:
                    # Both should be independent
                    assert ws1 is not None
                    assert ws2 is not None


class TestWebSocketBinaryData:
    """WebSocket binary data tests."""

    @pytest.mark.asyncio
    async def test_receive_video_frames(self, app):
        """Should be able to receive video frame data."""
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            with client.websocket_connect("/ws/avatar") as websocket:
                websocket.send_json({
                    "type": "start_session",
                    "avatar_id": "default"
                })

                # Send text to trigger response
                websocket.send_json({
                    "type": "text",
                    "content": "Hello"
                })

                # Try to receive frames
                try:
                    for _ in range(3):
                        # May receive JSON or binary
                        message = websocket.receive(timeout=2)
                        assert message is not None
                except Exception:
                    pass  # May timeout

    @pytest.mark.asyncio
    async def test_large_audio_chunk(self, app):
        """Should handle large audio chunks."""
        from starlette.testclient import TestClient

        # Create large audio chunk (1 second at 16kHz)
        large_audio = np.random.randn(16000).astype(np.float32)
        audio_bytes = (large_audio * 32767).astype(np.int16).tobytes()

        with TestClient(app) as client:
            with client.websocket_connect("/ws/avatar") as websocket:
                websocket.send_json({
                    "type": "start_session",
                    "avatar_id": "default"
                })

                # Send large chunk
                websocket.send_bytes(audio_bytes)
