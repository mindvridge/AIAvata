"""
WebSocket Handler for realtime avatar communication.

실시간 아바타 통신을 위한 WebSocket 핸들러
"""

import asyncio
import json
import logging
from typing import Optional
from uuid import UUID

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class AvatarWebSocketHandler:
    """
    아바타 WebSocket 통신 핸들러

    Protocol:
    - Client → Server: 오디오 청크 (bytes) 또는 제어 메시지 (JSON)
    - Server → Client: 비디오 프레임 (bytes) 또는 상태 메시지 (JSON)
    """

    def __init__(self, pipeline):
        """
        Initialize WebSocket Handler.

        Args:
            pipeline: PipelineOrchestrator 인스턴스
        """
        self.pipeline = pipeline
        self._active_connections: dict = {}

    async def handle_connection(
        self,
        websocket: WebSocket,
        session_id: Optional[str] = None,
    ):
        """
        WebSocket 연결 처리

        Args:
            websocket: WebSocket 연결
            session_id: 세션 ID (선택적)
        """
        await websocket.accept()
        connection_id = id(websocket)

        logger.info(f"WebSocket connected: {connection_id}")

        # 세션 ID 파싱
        uuid_session = None
        if session_id:
            try:
                uuid_session = UUID(session_id)
            except ValueError:
                await self._send_error(websocket, "Invalid session ID format")
                await websocket.close()
                return

        # 연결 등록
        self._active_connections[connection_id] = {
            "websocket": websocket,
            "session_id": uuid_session,
            "is_streaming": False,
        }

        try:
            # 초기 상태 전송
            await self._send_status(websocket, "connected")

            # 메시지 수신 루프
            while True:
                message = await websocket.receive()

                if message["type"] == "websocket.receive":
                    if "bytes" in message:
                        # 오디오 데이터 처리
                        await self._handle_audio(
                            websocket, message["bytes"], uuid_session
                        )
                    elif "text" in message:
                        # 제어 메시지 처리
                        await self._handle_control(
                            websocket, message["text"], uuid_session
                        )

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await self._send_error(websocket, str(e))

        finally:
            # 연결 해제
            if connection_id in self._active_connections:
                del self._active_connections[connection_id]

    async def _handle_audio(
        self,
        websocket: WebSocket,
        audio_data: bytes,
        session_id: Optional[UUID],
    ):
        """
        오디오 데이터 처리

        Args:
            websocket: WebSocket 연결
            audio_data: 오디오 바이트 데이터
            session_id: 세션 ID
        """
        connection_id = id(websocket)
        connection = self._active_connections.get(connection_id)

        if connection is None:
            return

        # 이미 스트리밍 중이면 무시 (중복 요청 방지)
        if connection["is_streaming"]:
            logger.warning("Already streaming, ignoring audio input")
            return

        connection["is_streaming"] = True

        try:
            # 상태 업데이트
            await self._send_status(websocket, "processing")

            # 파이프라인 처리 및 프레임 스트리밍
            async for frame in self.pipeline.process_audio_input(
                audio=audio_data,
                session_id=session_id,
            ):
                # 비디오 프레임 전송
                await websocket.send_bytes(frame.data)

            # 처리 완료
            await self._send_status(websocket, "idle")

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            await self._send_error(websocket, f"Processing error: {e}")

        finally:
            connection["is_streaming"] = False

    async def _handle_control(
        self,
        websocket: WebSocket,
        message_text: str,
        session_id: Optional[UUID],
    ):
        """
        제어 메시지 처리

        Args:
            websocket: WebSocket 연결
            message_text: JSON 메시지 텍스트
            session_id: 세션 ID
        """
        try:
            message = json.loads(message_text)
            msg_type = message.get("type")

            if msg_type == "ping":
                # Ping/Pong
                await self._send_json(websocket, {"type": "pong"})

            elif msg_type == "set_emotion":
                # 감정 변경
                emotion = message.get("emotion", "neutral")
                await self._handle_set_emotion(websocket, emotion, session_id)

            elif msg_type == "start_idle":
                # Idle 스트리밍 시작
                await self._handle_start_idle(websocket, session_id)

            elif msg_type == "stop":
                # 스트리밍 중지
                await self._send_status(websocket, "stopped")

            elif msg_type == "get_status":
                # 상태 조회
                await self._handle_get_status(websocket, session_id)

            else:
                await self._send_error(websocket, f"Unknown message type: {msg_type}")

        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON message")

    async def _handle_set_emotion(
        self,
        websocket: WebSocket,
        emotion_str: str,
        session_id: Optional[UUID],
    ):
        """감정 변경 처리"""
        from ..models.emotion import Emotion

        try:
            emotion = Emotion(emotion_str)
            self.pipeline.renderer.set_emotion(emotion)

            if session_id:
                session = self.pipeline.get_session(session_id)
                if session:
                    session.current_emotion = emotion

            await self._send_json(websocket, {
                "type": "emotion_changed",
                "emotion": emotion_str,
            })

        except ValueError:
            await self._send_error(websocket, f"Invalid emotion: {emotion_str}")

    async def _handle_start_idle(
        self,
        websocket: WebSocket,
        session_id: Optional[UUID],
    ):
        """Idle 스트리밍 시작"""
        connection_id = id(websocket)
        connection = self._active_connections.get(connection_id)

        if connection is None:
            return

        if connection["is_streaming"]:
            await self._send_error(websocket, "Already streaming")
            return

        connection["is_streaming"] = True

        try:
            await self._send_status(websocket, "idle_streaming")

            async for frame in self.pipeline.stream_idle(
                session_id=session_id,
                duration=30.0,  # 30초 후 자동 중지
            ):
                # 연결이 끊어지면 중지
                if connection_id not in self._active_connections:
                    break

                await websocket.send_bytes(frame.data)

        except Exception as e:
            logger.error(f"Idle streaming error: {e}")

        finally:
            if connection_id in self._active_connections:
                self._active_connections[connection_id]["is_streaming"] = False

    async def _handle_get_status(
        self,
        websocket: WebSocket,
        session_id: Optional[UUID],
    ):
        """상태 조회"""
        status = {
            "type": "status",
            "session_id": str(session_id) if session_id else None,
            "pipeline_ready": True,
        }

        if session_id:
            session = self.pipeline.get_session(session_id)
            if session:
                status.update({
                    "current_emotion": session.current_emotion.value,
                    "pipeline_state": session.pipeline_state.value,
                    "total_interactions": session.total_interactions,
                })

        await self._send_json(websocket, status)

    async def _send_status(self, websocket: WebSocket, status: str):
        """상태 메시지 전송"""
        await self._send_json(websocket, {"type": "status", "status": status})

    async def _send_error(self, websocket: WebSocket, error: str):
        """에러 메시지 전송"""
        await self._send_json(websocket, {"type": "error", "error": error})

    async def _send_json(self, websocket: WebSocket, data: dict):
        """JSON 메시지 전송"""
        try:
            await websocket.send_text(json.dumps(data))
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    def get_active_connections_count(self) -> int:
        """활성 연결 수 반환"""
        return len(self._active_connections)


# 전역 핸들러 인스턴스 (main.py에서 설정)
websocket_handler: Optional[AvatarWebSocketHandler] = None


def set_websocket_handler(pipeline):
    """WebSocket 핸들러 설정"""
    global websocket_handler
    websocket_handler = AvatarWebSocketHandler(pipeline)
    return websocket_handler


def get_websocket_handler() -> Optional[AvatarWebSocketHandler]:
    """WebSocket 핸들러 반환 (getter 함수)"""
    return websocket_handler