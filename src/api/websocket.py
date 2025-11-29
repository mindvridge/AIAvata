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

            # 백그라운드에서 idle 스트림 자동 시작
            idle_task = asyncio.create_task(
                self._start_idle_stream_background(websocket, uuid_session, connection_id)
            )

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
            # idle 스트림 취소
            if idle_task and not idle_task.done():
                idle_task.cancel()
                try:
                    await idle_task
                except asyncio.CancelledError:
                    pass

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

            elif msg_type == "chat":
                # 텍스트 채팅 메시지 처리
                text = message.get("text", "")
                if text:
                    await self._handle_chat(websocket, text, session_id)

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

    async def _start_idle_stream_background(
        self,
        websocket: WebSocket,
        session_id: Optional[UUID],
        connection_id: int,
    ):
        """백그라운드에서 idle 스트림 자동 시작 (연결 즉시)"""
        connection = self._active_connections.get(connection_id)
        if connection is None:
            return

        # 이미 스트리밍 중이면 건너뛰기
        if connection.get("is_streaming"):
            return

        connection["is_streaming"] = True

        try:
            logger.info(f"Starting auto idle stream for connection: {connection_id}")
            await self._send_status(websocket, "idle_streaming")

            frame_count = 0
            async for frame in self.pipeline.stream_idle(
                session_id=session_id,
                duration=-1,  # 무한 스트림 (연결이 끊어질 때까지)
            ):
                # 연결이 끊어지면 중지
                if connection_id not in self._active_connections:
                    logger.info(f"Connection {connection_id} closed, stopping idle stream")
                    break

                # 오디오 처리 중이면 idle 프레임 건너뛰기
                if self._active_connections[connection_id].get("processing_audio"):
                    continue

                try:
                    frame_count += 1
                    if frame_count % 30 == 0:  # 매 30프레임마다 로그
                        logger.debug(f"Sent {frame_count} idle frames to connection {connection_id}")
                    
                    await websocket.send_bytes(frame.data)
                except Exception as e:
                    logger.error(f"Error sending idle frame: {e}")
                    break

        except asyncio.CancelledError:
            logger.info(f"Idle stream cancelled for connection: {connection_id}")
        except Exception as e:
            logger.error(f"Auto idle streaming error: {e}")
        finally:
            if connection_id in self._active_connections:
                self._active_connections[connection_id]["is_streaming"] = False

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

    async def _handle_chat(
        self,
        websocket: WebSocket,
        text: str,
        session_id: Optional[UUID],
    ):
        """
        텍스트 채팅 메시지 처리
        
        Args:
            websocket: WebSocket 연결
            text: 사용자 입력 텍스트
            session_id: 세션 ID
        """
        connection_id = id(websocket)
        connection = self._active_connections.get(connection_id)

        if connection is None:
            return

        logger.info(f"Chat message received: {text[:50]}...")

        try:
            # 처리 상태 전송
            await self._send_json(websocket, {
                "type": "chat_status",
                "status": "processing",
                "user_message": text,
            })

            # 세션에서 시스템 프롬프트 가져오기
            system_prompt = "당신은 친절하고 공감능력이 뛰어난 AI 어시스턴트입니다. 사용자의 감정에 맞춰 대화해주세요."
            if session_id:
                session = self.pipeline.get_session(session_id)
                if session and hasattr(session, 'system_prompt') and session.system_prompt:
                    system_prompt = session.system_prompt

            # LLM 응답 생성 (올바른 메서드: generate)
            response_text = await self.pipeline.llm.generate(
                user_message=text,
                system_prompt=system_prompt,
            )

            logger.info(f"LLM response: {response_text[:50]}...")

            # 응답 텍스트 전송
            await self._send_json(websocket, {
                "type": "chat_response",
                "text": response_text,
                "user_message": text,
            })

            # TTS로 음성 생성 및 립싱크 아바타 렌더링
            try:
                await self._process_chat_with_tts(websocket, response_text, session_id)
            except Exception as e:
                logger.warning(f"TTS/Lipsync processing failed (non-critical): {e}")
                # TTS 실패해도 텍스트 응답은 이미 전송했으므로 계속 진행

        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            await self._send_json(websocket, {
                "type": "chat_error",
                "error": str(e),
                "user_message": text,
            })

    async def _process_chat_with_tts(
        self,
        websocket: WebSocket,
        response_text: str,
        session_id: Optional[UUID],
    ):
        """
        TTS로 음성 생성 및 립싱크 비디오 스트리밍
        
        Args:
            websocket: WebSocket 연결
            response_text: LLM 응답 텍스트
            session_id: 세션 ID
        """
        connection_id = id(websocket)
        connection = self._active_connections.get(connection_id)

        if connection is None:
            return

        # 이미 처리 중이면 건너뛰기
        if connection.get("is_streaming"):
            return

        connection["is_streaming"] = True
        connection["processing_audio"] = True

        try:
            logger.info("Generating TTS audio for chat response...")

            # TTS로 음성 생성 (numpy array 반환)
            audio_data_np = await self.pipeline.tts.generate(
                text=response_text,
                voice_id=None,  # 기본 음성 사용
            )

            if audio_data_np is not None and len(audio_data_np) > 0:
                # numpy array를 bytes로 변환 (16-bit PCM)
                audio_data_bytes = self.pipeline.tts._audio_to_bytes(audio_data_np)
                logger.info(f"TTS audio generated: {len(audio_data_np)} samples ({len(audio_data_bytes)} bytes)")

                # TTS 오디오를 프론트엔드로 전송 (파동 그래프용)
                import base64
                audio_base64 = base64.b64encode(audio_data_bytes).decode('utf-8')
                await self._send_json(websocket, {
                    "type": "audio_data",
                    "data": audio_base64,  # "audio" -> "data"로 변경 (프론트엔드와 일치)
                    "sample_rate": self.pipeline.tts.sample_rate,
                })

                # bytes를 AsyncGenerator로 변환
                async def audio_stream_generator():
                    # 오디오를 청크로 나누어 전송
                    chunk_size = 4096  # bytes
                    offset = 0
                    while offset < len(audio_data_bytes):
                        chunk = audio_data_bytes[offset:offset + chunk_size]
                        yield chunk
                        offset += chunk_size
                
                # 립싱크가 적용된 비디오 프레임 스트림 생성
                try:
                    logger.info(f"Starting lip sync rendering: sample_rate={self.pipeline.tts.sample_rate}, audio_bytes={len(audio_data_bytes)}")
                    frame_count = 0
                    async for frame in self.pipeline.renderer.render_with_audio(
                        audio_stream=audio_stream_generator(),
                        audio_sample_rate=self.pipeline.tts.sample_rate,
                    ):
                        # 연결이 끊어지면 중지
                        if connection_id not in self._active_connections:
                            logger.warning("Connection closed during lip sync streaming")
                            break

                        # 비디오 프레임 전송
                        await websocket.send_bytes(frame.data)
                        frame_count += 1
                        
                        if frame_count % 30 == 0:  # 30프레임마다 로그
                            logger.debug(f"Sent {frame_count} lip sync frames")

                    logger.info(f"Lipsync video stream completed: {frame_count} frames sent")
                except Exception as e:
                    logger.error(f"Error in render_with_audio: {e}", exc_info=True)
                    # 립싱크 실패해도 연결은 유지
            else:
                logger.warning("TTS audio generation returned empty data")

        except Exception as e:
            logger.error(f"TTS/Lipsync processing error: {e}")

        finally:
            if connection_id in self._active_connections:
                self._active_connections[connection_id]["processing_audio"] = False
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