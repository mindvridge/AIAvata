"""
WebSocket Handler for realtime avatar communication.

실시간 아바타 통신을 위한 WebSocket 핸들러
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable
from uuid import UUID

from fastapi import WebSocket, WebSocketDisconnect

from ..models.schemas import ConnectionState, CONNECTION_STATE_TRANSITIONS

logger = logging.getLogger(__name__)


class ConnectionStateMachine:
    """
    WebSocket 연결 상태 머신

    상태 전이를 관리하고 유효하지 않은 전이를 방지합니다.
    """

    def __init__(
        self,
        initial_state: ConnectionState = ConnectionState.CONNECTING,
        on_state_change: Optional[Callable[[ConnectionState, ConnectionState], None]] = None,
    ):
        self._state = initial_state
        self._on_state_change = on_state_change
        self._state_history: list[tuple[float, ConnectionState]] = [
            (time.time(), initial_state)
        ]

    @property
    def state(self) -> ConnectionState:
        """현재 상태 반환"""
        return self._state

    @property
    def state_history(self) -> list[tuple[float, ConnectionState]]:
        """상태 변경 이력 반환"""
        return self._state_history.copy()

    def can_transition_to(self, new_state: ConnectionState) -> bool:
        """주어진 상태로 전이 가능한지 확인"""
        valid_transitions = CONNECTION_STATE_TRANSITIONS.get(self._state, set())
        return new_state in valid_transitions

    def transition_to(self, new_state: ConnectionState, force: bool = False) -> bool:
        """
        새 상태로 전이

        Args:
            new_state: 전이할 상태
            force: True이면 유효성 검사 무시

        Returns:
            전이 성공 여부
        """
        if not force and not self.can_transition_to(new_state):
            logger.warning(
                f"Invalid state transition: {self._state.value} → {new_state.value}"
            )
            return False

        old_state = self._state
        self._state = new_state
        self._state_history.append((time.time(), new_state))

        # 이력 크기 제한 (최근 100개만 유지)
        if len(self._state_history) > 100:
            self._state_history = self._state_history[-100:]

        logger.debug(f"State transition: {old_state.value} → {new_state.value}")

        if self._on_state_change:
            try:
                self._on_state_change(old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

        return True

    def is_active(self) -> bool:
        """연결이 활성 상태인지 확인"""
        return self._state in {
            ConnectionState.CONNECTED,
            ConnectionState.IDLE_STREAMING,
            ConnectionState.PROCESSING,
            ConnectionState.SPEAKING,
        }

    def is_busy(self) -> bool:
        """처리 중인 상태인지 확인"""
        return self._state in {
            ConnectionState.PROCESSING,
            ConnectionState.SPEAKING,
        }


@dataclass
class DisconnectedSession:
    """연결 끊김 후 재연결을 위한 세션 상태 저장"""
    session_id: UUID
    disconnected_at: float
    audio_buffer: deque = field(default_factory=deque)
    audio_buffer_size: int = 0
    was_streaming: bool = False


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
        self._disconnected_sessions: dict[UUID, DisconnectedSession] = {}
        # 상수 캐싱 (매 호출마다 재계산 방지)
        self._max_audio_buffer_size: int = (
            pipeline.settings.audio_sample_rate * 2 *
            pipeline.settings.audio_buffer_max_seconds
        )
        self._heartbeat_interval: int = pipeline.settings.websocket_heartbeat_interval
        self._heartbeat_timeout: int = pipeline.settings.websocket_heartbeat_timeout
        self._reconnect_window: int = pipeline.settings.websocket_reconnect_window

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

        # 재연결 세션 확인 및 복구
        restored_state = None
        if uuid_session and uuid_session in self._disconnected_sessions:
            restored_state = self._disconnected_sessions.pop(uuid_session)
            elapsed = time.time() - restored_state.disconnected_at
            if elapsed <= self._reconnect_window:
                logger.info(f"Session {uuid_session} reconnected after {elapsed:.1f}s")
            else:
                logger.info(f"Session {uuid_session} reconnect window expired ({elapsed:.1f}s > {self._reconnect_window}s)")
                restored_state = None

        # 상태 머신 생성
        state_machine = ConnectionStateMachine(
            initial_state=ConnectionState.CONNECTING
        )

        # 연결 등록
        self._active_connections[connection_id] = {
            "websocket": websocket,
            "session_id": uuid_session,
            "state_machine": state_machine,
            "idle_task": None,
            "audio_buffer": restored_state.audio_buffer if restored_state else deque(),
            "audio_buffer_size": restored_state.audio_buffer_size if restored_state else 0,
            "stream_lock": asyncio.Lock(),  # 스트리밍 상태 동기화용 락
            "last_activity": time.time(),
            "last_pong": time.time(),
            "heartbeat_task": None,
        }

        try:
            # 연결 완료 상태로 전이
            state_machine.transition_to(ConnectionState.CONNECTED)

            # 초기 상태 전송 (재연결 시 restored 상태 포함)
            if restored_state:
                await self._send_json(websocket, {
                    "type": "status",
                    "status": "reconnected",
                    "connection_state": state_machine.state.value,
                    "session_id": str(uuid_session),
                    "buffered_audio_bytes": restored_state.audio_buffer_size,
                })
            else:
                await self._send_state_status(websocket, state_machine.state)

            # Heartbeat 태스크 시작
            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(websocket, connection_id)
            )
            self._active_connections[connection_id]["heartbeat_task"] = heartbeat_task

            # 백그라운드에서 idle 스트림 자동 시작
            idle_task = asyncio.create_task(
                self._start_idle_stream_background(websocket, uuid_session, connection_id)
            )
            self._active_connections[connection_id]["idle_task"] = idle_task

            # 메시지 수신 루프
            while True:
                message = await websocket.receive()

                # 활동 시간 업데이트
                if connection_id in self._active_connections:
                    self._active_connections[connection_id]["last_activity"] = time.time()

                logger.debug(f"Raw message received: type={message.get('type')}, keys={list(message.keys())}")

                if message["type"] == "websocket.receive":
                    if "bytes" in message:
                        audio_data = message["bytes"]
                        # 오디오 청크 크기 검증
                        if len(audio_data) > self.pipeline.settings.max_audio_chunk_size:
                            logger.warning(f"Audio chunk too large: {len(audio_data)} bytes")
                            await self._send_error(websocket, "Audio chunk too large")
                            continue
                        logger.debug(f"Received audio bytes: {len(audio_data)} bytes")
                        await self._handle_audio(websocket, audio_data, uuid_session)
                    elif "text" in message:
                        text_data = message["text"]
                        # 메시지 크기 검증
                        if len(text_data) > self.pipeline.settings.max_message_size_bytes:
                            logger.warning(f"Message too large: {len(text_data)} bytes")
                            await self._send_error(websocket, "Message too large")
                            continue
                        logger.debug(f"Received text message: {text_data[:100]}...")
                        await self._handle_control(websocket, text_data, uuid_session)
                    else:
                        logger.warning(f"Unknown message format: {message}")

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await self._send_error(websocket, str(e))

        finally:
            # Heartbeat 태스크 취소
            connection = self._active_connections.get(connection_id)
            if connection and connection.get("heartbeat_task"):
                heartbeat_task = connection["heartbeat_task"]
                if not heartbeat_task.done():
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass

            # idle 스트림 취소
            if idle_task and not idle_task.done():
                idle_task.cancel()
                try:
                    await idle_task
                except asyncio.CancelledError:
                    pass

            # 세션 상태 저장 (재연결 지원)
            if connection_id in self._active_connections:
                conn = self._active_connections[connection_id]
                state_machine: ConnectionStateMachine = conn.get("state_machine")

                # DISCONNECTED 상태로 전이
                if state_machine:
                    state_machine.transition_to(ConnectionState.DISCONNECTED, force=True)

                if conn.get("session_id"):
                    # 만료된 세션 정리
                    self._cleanup_expired_disconnected_sessions()
                    # 현재 세션 상태 저장
                    self._disconnected_sessions[conn["session_id"]] = DisconnectedSession(
                        session_id=conn["session_id"],
                        disconnected_at=time.time(),
                        audio_buffer=conn.get("audio_buffer", deque()),
                        audio_buffer_size=conn.get("audio_buffer_size", 0),
                        was_streaming=state_machine.is_busy() if state_machine else False,
                    )
                    logger.info(f"Session {conn['session_id']} state saved for reconnection")
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

        state_machine: ConnectionStateMachine = connection["state_machine"]
        stream_lock = connection["stream_lock"]

        # 락을 사용하여 상태를 atomic하게 확인/전이
        async with stream_lock:
            # 이미 처리 중이면 버퍼링
            if state_machine.is_busy():
                self._buffer_audio(connection, audio_data)
                logger.debug(f"Audio buffered: {len(audio_data)} bytes (total: {connection['audio_buffer_size']} bytes)")
                return
            # PROCESSING 상태로 전이
            if not state_machine.transition_to(ConnectionState.PROCESSING):
                logger.warning("Failed to transition to PROCESSING state")
                return

        try:
            # 상태 업데이트 전송
            await self._send_state_status(websocket, state_machine.state)

            # 파이프라인 처리 및 프레임 스트리밍
            async for frame in self.pipeline.process_audio_input(
                audio=audio_data,
                session_id=session_id,
            ):
                # 비디오 프레임 전송
                await websocket.send_bytes(frame.data)

            # 처리 완료 후 버퍼에 있는 오디오 처리
            await self._process_buffered_audio(websocket, connection, session_id)

            # IDLE_STREAMING 상태로 복귀
            state_machine.transition_to(ConnectionState.IDLE_STREAMING)
            await self._send_state_status(websocket, state_machine.state)

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            state_machine.transition_to(ConnectionState.ERROR)
            await self._send_error(websocket, f"Processing error: {e}")
            # 에러 후 CONNECTED 상태로 복구
            state_machine.transition_to(ConnectionState.CONNECTED)

    def _buffer_audio(self, connection: dict, audio_data: bytes) -> None:
        """오디오 데이터를 버퍼에 추가 (최대 크기 제한 적용)"""
        audio_buffer = connection["audio_buffer"]
        buffer_size = connection["audio_buffer_size"]

        # 최대 버퍼 크기 초과 시 오래된 데이터 제거
        while buffer_size + len(audio_data) > self._max_audio_buffer_size and audio_buffer:
            removed = audio_buffer.popleft()
            buffer_size -= len(removed)
            logger.debug(f"Audio buffer overflow, removed {len(removed)} bytes")

        audio_buffer.append(audio_data)
        connection["audio_buffer_size"] = buffer_size + len(audio_data)

    async def _process_buffered_audio(
        self,
        websocket: WebSocket,
        connection: dict,
        session_id: Optional[UUID],
    ) -> None:
        """버퍼에 저장된 오디오 처리"""
        audio_buffer = connection["audio_buffer"]

        if not audio_buffer:
            return

        # 버퍼에서 모든 오디오 결합
        combined_audio = b"".join(audio_buffer)
        audio_buffer.clear()
        connection["audio_buffer_size"] = 0

        logger.info(f"Processing buffered audio: {len(combined_audio)} bytes")

        # 버퍼링된 오디오 처리
        try:
            async for frame in self.pipeline.process_audio_input(
                audio=combined_audio,
                session_id=session_id,
            ):
                await websocket.send_bytes(frame.data)
        except Exception as e:
            logger.error(f"Buffered audio processing error: {e}")

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
            logger.debug(f"Received control message: {message_text[:100]}...")
            message = json.loads(message_text)
            msg_type = message.get("type")

            logger.debug(f"Processing control message type: {msg_type}")

            if msg_type == "ping":
                # Ping/Pong (클라이언트 요청)
                await self._send_json(websocket, {"type": "pong"})

            elif msg_type == "pong":
                # Pong 응답 (heartbeat에 대한 클라이언트 응답)
                connection_id = id(websocket)
                if connection_id in self._active_connections:
                    self._active_connections[connection_id]["last_pong"] = time.time()
                    logger.debug(f"Pong received from connection {connection_id}")

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
                # 텍스트 길이 검증
                if len(text) > self.pipeline.settings.max_text_length:
                    logger.warning(f"Chat text too long: {len(text)} chars")
                    await self._send_error(websocket, f"Text too long (max {self.pipeline.settings.max_text_length} chars)")
                elif text:
                    logger.debug(f"Chat message received: '{text[:50]}...'")
                    await self._handle_chat(websocket, text, session_id)
                else:
                    logger.warning("Chat message text is empty")

            else:
                logger.warning(f"Unknown message type: {msg_type}")
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

        state_machine: ConnectionStateMachine = connection["state_machine"]

        # 이미 처리 중이면 건너뛰기
        if state_machine.is_busy():
            logger.debug(f"Idle stream skipped: state={state_machine.state.value}")
            return

        # IDLE_STREAMING 상태로 전이
        if not state_machine.transition_to(ConnectionState.IDLE_STREAMING):
            logger.debug(f"Cannot transition to IDLE_STREAMING from {state_machine.state.value}")
            return

        try:
            logger.info(f"Starting auto idle stream for connection: {connection_id}")
            await self._send_state_status(websocket, state_machine.state)

            frame_count = 0
            async for frame in self.pipeline.stream_idle(
                session_id=session_id,
                duration=-1,  # 무한 스트림 (연결이 끊어질 때까지)
            ):
                # 연결이 끊어지면 중지
                if connection_id not in self._active_connections:
                    logger.info(f"Connection {connection_id} closed, stopping idle stream")
                    break

                # 처리 중 상태로 변경되면 idle 프레임 건너뛰기
                if state_machine.is_busy():
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
                # CONNECTED 상태로 복귀
                state_machine.transition_to(ConnectionState.CONNECTED, force=True)

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

        state_machine: ConnectionStateMachine = connection["state_machine"]

        # 이미 처리 중이면 에러
        if state_machine.is_busy():
            await self._send_error(websocket, "Already streaming")
            return

        # IDLE_STREAMING 상태로 전이
        if not state_machine.transition_to(ConnectionState.IDLE_STREAMING):
            await self._send_error(websocket, f"Cannot start idle from {state_machine.state.value}")
            return

        try:
            await self._send_state_status(websocket, state_machine.state)

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
                state_machine.transition_to(ConnectionState.CONNECTED, force=True)

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

        state_machine: ConnectionStateMachine = connection["state_machine"]
        logger.info(f"Chat message received: {text[:50]}...")

        # PROCESSING 상태로 전이
        state_machine.transition_to(ConnectionState.PROCESSING)

        # 세션 가져오기
        session = None
        if session_id:
            session = self.pipeline.get_session(session_id)

        try:
            # 처리 상태 전송
            await self._send_json(websocket, {
                "type": "chat_status",
                "status": "processing",
                "user_message": text,
            })

            # 대화 히스토리에 사용자 메시지 추가
            if session:
                session.conversation_history.append({
                    "role": "user",
                    "content": text,
                })
                logger.debug(f"Added user message to history: {text[:50]}...")

            # 세션에서 시스템 프롬프트 가져오기
            system_prompt = self.pipeline.settings.system_prompt

            # 대화 히스토리 전달 (현재 메시지 제외)
            conversation_history = session.conversation_history[:-1] if session else []

            # LLM 응답 생성 (올바른 메서드: generate)
            response_text = await self.pipeline.llm.generate(
                user_message=text,
                system_prompt=system_prompt,
                conversation_history=conversation_history,
            )

            logger.info(f"LLM response: {response_text[:50]}...")

            # 대화 히스토리에 어시스턴트 응답 추가
            if session:
                session.conversation_history.append({
                    "role": "assistant",
                    "content": response_text,
                })
                logger.debug(f"Added assistant response to history: {response_text[:50]}...")

            # TTS로 음성 생성 및 립싱크 아바타 렌더링
            # 텍스트와 오디오를 동시에 전송하여 동기화
            logger.info("🎤 Starting TTS and lip sync processing for chat response...")
            try:
                await self._process_chat_with_tts(websocket, response_text, session_id, text)
                logger.info("✅ TTS and lip sync processing completed successfully")
            except Exception as e:
                logger.error(f"❌ TTS/Lipsync processing failed: {e}", exc_info=True)
                # TTS 실패 시에만 텍스트 응답 전송 (폴백)
                await self._send_json(websocket, {
                    "type": "chat_response",
                    "text": response_text,
                    "user_message": text,
                })

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
        user_message: str = "",
    ):
        """
        TTS로 음성 생성 및 립싱크 비디오 스트리밍
        텍스트 응답과 오디오를 동시에 전송하여 동기화

        Args:
            websocket: WebSocket 연결
            response_text: LLM 응답 텍스트
            session_id: 세션 ID
            user_message: 사용자 원본 메시지
        """
        connection_id = id(websocket)
        connection = self._active_connections.get(connection_id)

        if connection is None:
            logger.error("Connection not found, cannot process TTS")
            return

        state_machine: ConnectionStateMachine = connection["state_machine"]
        logger.debug(f"Processing TTS for response: '{response_text[:50]}...'")

        # idle 스트림 중지 (채팅 처리 시작 전)
        if connection.get("idle_task"):
            idle_task = connection["idle_task"]
            if idle_task and not idle_task.done():
                logger.debug("Stopping idle stream for chat processing")
                idle_task.cancel()
                try:
                    await asyncio.wait_for(idle_task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                connection["idle_task"] = None

        # SPEAKING 상태로 전이
        state_machine.transition_to(ConnectionState.SPEAKING)

        try:

            # TTS로 음성 생성 (numpy array 반환)
            audio_data_np = await self.pipeline.tts.synthesize(
                text=response_text,
                voice_id=None,  # 기본 음성 사용
            )
            
            logger.debug(f"TTS generation result: samples={len(audio_data_np) if audio_data_np is not None else 0}")

            if audio_data_np is None or len(audio_data_np) == 0:
                logger.warning("TTS returned empty audio")
                return

            # numpy array를 bytes로 변환 (16-bit PCM)
            audio_data_bytes = self.pipeline.tts._audio_to_bytes(audio_data_np)
            logger.debug(f"TTS audio generated: {len(audio_data_np)} samples ({len(audio_data_bytes)} bytes)")

            # TTS 오디오와 텍스트 응답을 동시에 전송 (동기화)
            import base64
            audio_base64 = base64.b64encode(audio_data_bytes).decode('utf-8')

            # 텍스트 응답 먼저 전송
            await self._send_json(websocket, {
                "type": "chat_response",
                "text": response_text,
                "user_message": user_message,
            })

            # 오디오 데이터 바로 이어서 전송
            await self._send_json(websocket, {
                "type": "audio_data",
                "data": audio_base64,
                "sample_rate": self.pipeline.tts.sample_rate,
            })
            logger.debug("Sent chat_response and audio_data together for synchronization")

            # bytes를 AsyncGenerator로 변환
            async def audio_stream_generator():
                # 오디오를 청크로 나누어 전송
                chunk_size = self.pipeline.settings.tts_chunk_size
                offset = 0
                while offset < len(audio_data_bytes):
                    chunk = audio_data_bytes[offset:offset + chunk_size]
                    yield chunk
                    offset += chunk_size

            # 립싱크가 적용된 비디오 프레임 스트림 생성
            try:
                logger.debug(f"Starting lip sync rendering: sample_rate={self.pipeline.tts.sample_rate}, audio_bytes={len(audio_data_bytes)}")
                await self._send_status(websocket, "speaking")

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

                logger.debug(f"Lipsync video stream completed: {frame_count} frames sent")

                # 립싱크 완료 후 CONNECTED 상태로 전이 및 idle 스트림 재시작
                if connection_id in self._active_connections:
                    state_machine.transition_to(ConnectionState.CONNECTED)
                    await self._send_state_status(websocket, state_machine.state)

                    connection = self._active_connections[connection_id]
                    if not connection.get("idle_task") or (hasattr(connection["idle_task"], 'done') and connection["idle_task"].done()):
                        logger.debug("Restarting idle stream after lip sync")
                        idle_task = asyncio.create_task(
                            self._start_idle_stream_background(websocket, session_id, connection_id)
                        )
                        connection["idle_task"] = idle_task
            except Exception as e:
                logger.error(f"Error in render_with_audio: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"TTS/Lipsync processing error: {e}")
            state_machine.transition_to(ConnectionState.ERROR)

        finally:
            if connection_id in self._active_connections:
                # 에러 상태가 아니면 CONNECTED로 복귀
                if state_machine.state == ConnectionState.ERROR:
                    state_machine.transition_to(ConnectionState.CONNECTED)

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
        """상태 메시지 전송 (레거시 호환)"""
        await self._send_json(websocket, {"type": "status", "status": status})

    async def _send_state_status(self, websocket: WebSocket, state: ConnectionState):
        """상태 머신 상태 메시지 전송"""
        await self._send_json(websocket, {
            "type": "status",
            "status": state.value,
            "connection_state": state.value,
        })

    async def _send_error(self, websocket: WebSocket, error: str):
        """에러 메시지 전송"""
        await self._send_json(websocket, {"type": "error", "error": error})

    async def _send_json(self, websocket: WebSocket, data: dict):
        """JSON 메시지 전송"""
        try:
            await websocket.send_text(json.dumps(data))
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    async def _heartbeat_loop(self, websocket: WebSocket, connection_id: int):
        """
        주기적으로 ping을 보내고 응답을 확인하는 heartbeat 루프

        Args:
            websocket: WebSocket 연결
            connection_id: 연결 ID
        """
        try:
            while connection_id in self._active_connections:
                await asyncio.sleep(self._heartbeat_interval)

                connection = self._active_connections.get(connection_id)
                if connection is None:
                    break

                # Ping 전송
                try:
                    await self._send_json(websocket, {"type": "ping"})
                except Exception as e:
                    logger.warning(f"Failed to send heartbeat ping: {e}")
                    break

                # 타임아웃 대기 후 pong 응답 확인
                await asyncio.sleep(self._heartbeat_timeout)

                connection = self._active_connections.get(connection_id)
                if connection is None:
                    break

                # pong 응답 확인
                time_since_pong = time.time() - connection["last_pong"]
                if time_since_pong > self._heartbeat_interval + self._heartbeat_timeout:
                    logger.warning(
                        f"Connection {connection_id} heartbeat timeout "
                        f"(no pong for {time_since_pong:.1f}s)"
                    )
                    # 클라이언트에 연결 끊김 알림 시도
                    try:
                        await self._send_json(websocket, {
                            "type": "error",
                            "error": "heartbeat_timeout",
                            "message": "Connection timeout - please reconnect",
                        })
                        await websocket.close(code=1001, reason="Heartbeat timeout")
                    except Exception:
                        pass
                    break

        except asyncio.CancelledError:
            logger.debug(f"Heartbeat loop cancelled for connection {connection_id}")
        except Exception as e:
            logger.error(f"Heartbeat loop error: {e}")

    def _cleanup_expired_disconnected_sessions(self):
        """만료된 disconnected 세션 정리"""
        now = time.time()
        expired_sessions = [
            session_id
            for session_id, session in self._disconnected_sessions.items()
            if now - session.disconnected_at > self._reconnect_window
        ]
        for session_id in expired_sessions:
            del self._disconnected_sessions[session_id]
            logger.debug(f"Expired disconnected session cleaned up: {session_id}")

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired disconnected sessions")

    def get_active_connections_count(self) -> int:
        """활성 연결 수 반환"""
        return len(self._active_connections)

    def get_disconnected_sessions_count(self) -> int:
        """재연결 대기 중인 세션 수 반환"""
        return len(self._disconnected_sessions)


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