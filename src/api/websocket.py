"""
WebSocket Handler for realtime avatar communication.

실시간 아바타 통신을 위한 WebSocket 핸들러
"""

# 🔑 모듈 로드 확인 (이 메시지가 나오면 최신 코드가 로드된 것)
print("=" * 60, flush=True)
print("[WEBSOCKET MODULE] 최신 코드 로드됨 - 2026-01-01-v3", flush=True)
print("=" * 60, flush=True)

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
from .routes import get_initialization_complete

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 🔑 로거 레벨 명시적 설정 (INFO 보장)


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
        # 🔑 WebSocket 연결 시 로깅 설정 재확인 (uvicorn이 로깅을 덮어쓴 경우 대비)
        import logging
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        # uvicorn 로거들 재설정
        for uvicorn_logger_name in ['uvicorn', 'uvicorn.access', 'uvicorn.error']:
            uvicorn_logger = logging.getLogger(uvicorn_logger_name)
            uvicorn_logger.handlers = []
            uvicorn_logger.setLevel(logging.INFO)
            uvicorn_logger.propagate = True
        # 주요 모듈 로거 재설정
        for module_name in ['src', 'src.api', 'src.api.websocket']:
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(logging.INFO)
            module_logger.propagate = True
        
        # 🔑 테스트: 함수 진입 확인
        print("=" * 60, flush=True)
        print(f"[TEST] handle_connection called! session_id={session_id}", flush=True)
        print("=" * 60, flush=True)
        logger.info("=" * 60)
        logger.info(f"🔧 WebSocket 연결 시 로깅 설정 재확인 완료 - session_id={session_id}")
        logger.info("=" * 60)
        
        # 🔑 초기화 완료 여부 확인 (연결 수락 전에 확인)
        if not get_initialization_complete():
            logger.warning(f"⚠️ WebSocket 연결 거부: 서비스가 아직 초기화 중입니다 (session_id={session_id})")
            await websocket.close(code=1013, reason="Service initializing, please wait")
            return
        
        await websocket.accept()
        connection_id = id(websocket)

        logger.info("=" * 60)
        logger.info(f"🔌 WebSocket 연결됨: connection_id={connection_id}")
        logger.info("=" * 60)

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

            # 백그라운드에서 idle 스트림 자동 시작 (설정에 따라 비활성화 가능)
            # 기본값은 True (로컬 비디오 재생 사용)
            if not getattr(self.pipeline.settings, 'disable_server_idle_stream', True):
                idle_task = asyncio.create_task(
                    self._start_idle_stream_background(websocket, uuid_session, connection_id)
                )
                self._active_connections[connection_id]["idle_task"] = idle_task
                logger.info(f"Server-side idle stream enabled for connection {connection_id}")
            else:
                logger.info(f"✅ Server-side idle stream disabled for connection {connection_id} (using local video playback in frontend)")
                self._active_connections[connection_id]["idle_task"] = None

            # 메시지 수신 루프
            logger.info("🔄 메시지 수신 루프 시작")
            print("[TEST] 메시지 수신 루프 시작", flush=True)  # 🔑 테스트 출력
            while True:
                message = await websocket.receive()
                print(f"[TEST] 메시지 수신됨: type={message.get('type')}", flush=True)  # 🔑 테스트 출력

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
                        logger.info(f"📥 오디오 데이터 수신: {len(audio_data)} bytes")
                        print(f"[TEST] 오디오 데이터 수신: {len(audio_data)} bytes", flush=True)  # 🔑 테스트 출력
                        await self._handle_audio(websocket, audio_data, uuid_session)
                    elif "text" in message:
                        text_data = message["text"]
                        logger.info(f"📥 텍스트 메시지 수신: {text_data}")
                        print(f"[TEST] 텍스트 메시지 수신: {text_data}", flush=True)  # 🔑 테스트 출력
                        # 메시지 크기 검증
                        if len(text_data) > self.pipeline.settings.max_message_size_bytes:
                            logger.warning(f"Message too large: {len(text_data)} bytes")
                            await self._send_error(websocket, "Message too large")
                            continue
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

            # idle 스트림 취소 (connection에서 가져옴)
            if connection and connection.get("idle_task"):
                idle_task = connection["idle_task"]
                if not idle_task.done():
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
            frame_count = 0
            async for frame in self.pipeline.process_audio_input(
                audio=combined_audio,
                session_id=session_id,
            ):
                # 🔑 첫 프레임 크기 로깅 (512x512 문제 디버그)
                if frame_count == 0:
                    logger.info(f"📐 [WebSocket Buffered Audio] 첫 번째 프레임: {frame.width}x{frame.height}, JPEG bytes: {len(frame.data)}")
                frame_count += 1
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
                    logger.info(f"💬 채팅 메시지 수신 (JSON 파싱 후): {text}")
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
                # 🔑 첫 프레임 크기 로깅 및 파일 저장 (512x512 문제 디버그)
                if frame_count == 0:
                    print(f"[FIRST FRAME] Width={frame.width}, Height={frame.height}, Bytes={len(frame.data)}", flush=True)
                    logger.info(f"📐 [WebSocket Idle] 첫 번째 프레임: {frame.width}x{frame.height}, JPEG bytes: {len(frame.data)}")
                    # 🔑 512x512 프레임 감지 시 즉시 경고
                    if frame.width == 512 and frame.height == 512:
                        print(f"[ERROR] 첫 프레임이 512x512입니다! 백엔드 프레임 생성 문제!", flush=True)
                        logger.error(f"❌ 첫 프레임이 512x512입니다! 백엔드 프레임 생성 문제!")
                    # 🔑 디버그: 첫 번째 프레임을 파일로 저장
                    try:
                        import os
                        debug_path = os.path.abspath("debug_first_frame_idle.jpg")
                        logger.info(f"🔍 디버그 파일 저장 시도: {debug_path}")
                        print(f"[DEBUG] 디버그 파일 저장: {debug_path}", flush=True)
                        with open(debug_path, "wb") as f:
                            f.write(frame.data)
                        file_size = os.path.getsize(debug_path)
                        logger.info(f"✅ 디버그: 첫 번째 프레임 저장 성공 → {debug_path} ({file_size} bytes)")
                        print(f"[DEBUG] 파일 저장 성공: {file_size} bytes", flush=True)
                    except Exception as e:
                        import traceback
                        logger.error(f"❌ 디버그 파일 저장 실패: {e}")
                        logger.error(f"❌ 에러 상세: {traceback.format_exc()}")
                        print(f"[ERROR] 파일 저장 실패: {e}", flush=True)
                # 연결이 끊어지면 중지
                if connection_id not in self._active_connections:
                    logger.info(f"Connection {connection_id} closed, stopping idle stream")
                    break

                # SPEAKING 상태일 때만 프레임 전송 건너뛰기
                # (TTS 생성 중에는 idle 루프 계속 재생)
                if state_machine.state == ConnectionState.SPEAKING:
                    continue

                try:
                    frame_count += 1

                    # 🔑 WebSocket 전송 직전 최종 프레임 크기 검증 및 강제 리사이즈
                    # JPEG 디코딩 → 크기 확인 → 필요시 리사이즈 → 재인코딩
                    frame_data = frame.data
                    import cv2
                    import numpy as np

                    # JPEG 디코딩하여 실제 크기 확인
                    np_arr = np.frombuffer(frame_data, np.uint8)
                    decoded_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    if decoded_frame is not None:
                        actual_h, actual_w = decoded_frame.shape[:2]
                        target_w, target_h = 784, 1176  # 원본 비디오 크기 (avata_ani.mp4)

                        # 🔑 512x512 또는 잘못된 크기 감지 시 강제 리사이즈
                        if actual_w != target_w or actual_h != target_h:
                            if frame_count <= 3 or frame_count % 30 == 0:
                                print(f"[WEBSOCKET RESIZE] Frame #{frame_count}: {actual_w}x{actual_h} → {target_w}x{target_h}", flush=True)
                                logger.warning(f"⚠️ [WebSocket] 프레임 크기 불일치 감지: {actual_w}x{actual_h} → {target_w}x{target_h}로 강제 리사이즈")

                            # 강제 리사이즈
                            resized_frame = cv2.resize(decoded_frame, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

                            # JPEG 재인코딩
                            _, encoded = cv2.imencode(".jpg", resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            frame_data = encoded.tobytes()

                            if frame_count <= 3:
                                print(f"[WEBSOCKET RESIZE DONE] Frame #{frame_count}: Now {target_w}x{target_h}, {len(frame_data)} bytes", flush=True)
                        elif frame_count <= 3:
                            print(f"[FRAME #{frame_count}] OK: {actual_w}x{actual_h}, {len(frame_data)} bytes", flush=True)
                    else:
                        # 디코딩 실패 시 원본 사용
                        if frame_count <= 3:
                            print(f"[WARNING] Frame #{frame_count}: JPEG decode failed, sending original", flush=True)

                    if frame_count <= 3:
                        logger.info(f"🎬 Idle frame #{frame_count}: sending {len(frame_data)} bytes")
                    elif frame_count % 30 == 0:
                        logger.info(f"🎬 Sent {frame_count} idle frames to connection {connection_id}")

                    await websocket.send_bytes(frame_data)
                except Exception as e:
                    logger.error(f"Error sending idle frame: {e}")
                    break

        except asyncio.CancelledError:
            logger.info(f"Idle stream cancelled for connection: {connection_id}")
        except Exception as e:
            logger.error(f"Auto idle streaming error: {e}")
        finally:
            if connection_id in self._active_connections:
                # busy 상태(PROCESSING, SPEAKING)가 아닐 때만 CONNECTED로 복귀
                # 채팅 처리 중 idle 취소 시 상태 덮어쓰기 방지
                if not state_machine.is_busy():
                    state_machine.transition_to(ConnectionState.CONNECTED, force=True)
                else:
                    logger.debug(f"Skipping CONNECTED transition: already in {state_machine.state.value}")

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
                # busy 상태가 아닐 때만 CONNECTED로 복귀
                if not state_machine.is_busy():
                    state_machine.transition_to(ConnectionState.CONNECTED, force=True)

    def _detect_language(self, text: str) -> str:
        """
        텍스트에서 언어 감지 (간단한 휴리스틱)

        Args:
            text: 분석할 텍스트

        Returns:
            언어 코드 (ko, en, ja, zh)
        """
        # 한글 포함 여부
        if any('\uac00' <= c <= '\ud7a3' for c in text):
            return "ko"
        # 일본어 히라가나/가타카나 포함 여부
        if any('\u3040' <= c <= '\u30ff' for c in text):
            return "ja"
        # 중국어 간체/번체 포함 여부 (한글 제외)
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            return "zh"
        # 기본값: 영어
        return "en"

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
        logger.info("=" * 60)
        logger.info(f"💬 사용자 메시지: {text}")
        logger.info("=" * 60)

        # 🎬 idle 루프는 계속 재생 - TTS/립싱크 준비 완료까지 PROCESSING 전이 안 함
        # _process_chat_with_tts에서 프레임 준비 완료 후 SPEAKING으로 전이

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

            logger.info(f"🤖 LLM 응답: {response_text}")

            # 대화 히스토리에 어시스턴트 응답 추가
            if session:
                session.conversation_history.append({
                    "role": "assistant",
                    "content": response_text,
                })
                logger.debug(f"Added assistant response to history: {response_text[:50]}...")

            # 언어 감지 (응답 텍스트 기준)
            detected_language = self._detect_language(response_text)
            logger.debug(f"Detected language: {detected_language}")

            # TTS로 음성 생성 및 립싱크 아바타 렌더링
            # 텍스트와 오디오를 동시에 전송하여 동기화
            logger.info("🎤 Starting TTS and lip sync processing for chat response...")
            try:
                await self._process_chat_with_tts(
                    websocket, response_text, session_id, text,
                    language=detected_language
                )
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

    @staticmethod
    def _make_audio_stream_generator(audio_bytes: bytes, chunk_size: int):
        """
        오디오 스트림 제너레이터 팩토리 함수 (클로저 버그 방지)

        Args:
            audio_bytes: 오디오 바이트 데이터
            chunk_size: 청크 크기

        Returns:
            비동기 제너레이터 함수
        """
        async def generator():
            offset = 0
            while offset < len(audio_bytes):
                chunk = audio_bytes[offset:offset + chunk_size]
                yield chunk
                offset += chunk_size
        return generator

    async def _process_chat_with_tts(
        self,
        websocket: WebSocket,
        response_text: str,
        session_id: Optional[UUID],
        user_message: str = "",
        language: str = "ko",
    ):
        """
        실시간 TTS 스트리밍으로 음성 생성 및 립싱크 비디오 전송

        문장별로 TTS를 생성하여 즉시 전송하므로 첫 응답 지연을 3-5초로 줄임
        (기존 전체 생성 방식은 50초+ 대기 필요)

        Args:
            websocket: WebSocket 연결
            response_text: LLM 응답 텍스트
            session_id: 세션 ID
            user_message: 사용자 원본 메시지
            language: 언어 코드 (ko, en, ja, zh)
        """
        import base64

        connection_id = id(websocket)
        connection = self._active_connections.get(connection_id)

        if connection is None:
            logger.error("Connection not found, cannot process TTS")
            return

        state_machine: ConnectionStateMachine = connection["state_machine"]
        logger.info(f"🎙️ 실시간 TTS 스트리밍 시작 (lang={language}): {response_text}")

        # 텍스트 응답 먼저 전송 (UI 업데이트용)
        await self._send_json(websocket, {
            "type": "chat_response",
            "text": response_text,
            "user_message": user_message,
        })

        await self._send_status(websocket, "processing")

        total_frames_sent = 0
        target_fps = self.pipeline.settings.target_fps
        frame_duration = 1.0 / target_fps
        chunk_size = self.pipeline.settings.tts_chunk_size

        try:
            # 문장 단위 TTS 스트리밍 (오디오-비디오 동기화를 위해)
            logger.info(f"🔄 문장별 TTS 스트리밍 시작: {response_text}")
            stream_started = False
            first_sentence = True
            
            # 문장별 실시간 TTS 스트리밍
            async for audio_np, sentence, idx, total in self.pipeline.tts.synthesize_sentences_streaming(
                text=response_text,
                voice_id=None,
                language=language,
            ):
                stream_started = True
                # 연결 확인
                if connection_id not in self._active_connections:
                    logger.warning("Connection closed during streaming")
                    break

                if audio_np is None or len(audio_np) == 0:
                    logger.warning(f"⚠️ 문장 {idx+1}/{total} 빈 오디오, 건너뜀")
                    continue
                
                logger.info(f"✅ 문장 {idx+1}/{total} 오디오 수신: {len(audio_np)} samples")
                
                # numpy array를 bytes로 변환 (16-bit PCM)
                audio_bytes = self.pipeline.tts._audio_to_bytes(audio_np)
                
                # 립싱크 비디오 프레임 생성 (오디오와 동기화)
                audio_generator = self._make_audio_stream_generator(audio_bytes, chunk_size)
                
                video_frames = []
                first_frame_logged = False
                print(f"[LIPSYNC START] 립싱크 프레임 생성 시작...", flush=True)
                try:
                    async for frame in self.pipeline.renderer.render_with_audio(
                        audio_stream=audio_generator(),
                        audio_sample_rate=self.pipeline.tts.sample_rate,
                    ):
                        # 🔑 첫 번째 프레임: JPEG 실제 크기 확인
                        if not first_frame_logged:
                            first_frame_logged = True
                            import cv2
                            import numpy as np

                            # JPEG 디코딩하여 실제 크기 확인
                            np_arr = np.frombuffer(frame.data, np.uint8)
                            decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                            if decoded is not None:
                                actual_h, actual_w = decoded.shape[:2]
                                print(f"[LIPSYNC FIRST FRAME] VideoFrame: {frame.width}x{frame.height}, JPEG 실제 크기: {actual_w}x{actual_h}", flush=True)
                                logger.info(f"📐 [LipSync 첫 프레임] VideoFrame: {frame.width}x{frame.height}, JPEG 실제 크기: {actual_w}x{actual_h}")

                                # 512x512 감지
                                if actual_w == 512 and actual_h == 512:
                                    print(f"[ERROR] ❌ 512x512 프레임 감지! MuseTalk에서 잘못된 크기 생성!", flush=True)
                                    logger.error(f"❌ 512x512 프레임 감지! MuseTalk에서 잘못된 크기 생성!")
                            else:
                                print(f"[WARNING] JPEG 디코딩 실패", flush=True)

                            # 디버그 파일 저장
                            try:
                                import os
                                debug_path = os.path.abspath("debug_first_frame_lipsync.jpg")
                                with open(debug_path, "wb") as f:
                                    f.write(frame.data)
                                print(f"[DEBUG] 첫 프레임 저장: {debug_path}", flush=True)
                            except Exception as e:
                                print(f"[ERROR] 파일 저장 실패: {e}", flush=True)

                        video_frames.append(frame.data)
                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    logger.error(f"❌ 문장 {idx+1}/{total} 립싱크 생성 실패: {str(e)}")
                    logger.error(f"❌ 에러 상세:\n{error_trace}")
                    continue

                logger.info(f"🎬 문장 {idx+1}/{total} 립싱크 완료: {len(video_frames)} 프레임")
                
                # 첫 문장일 때만 idle 루프 종료 대기 및 상태 전이
                if first_sentence:
                    first_sentence = False
                    
                    # 루프 끝 대기 (최대 5초)
                    logger.info("🎬 Waiting for idle loop to complete for smooth transition...")
                    loop_completed = await self.pipeline.renderer.wait_for_loop_end(timeout=5.5)
                    if loop_completed:
                        logger.info("✅ Idle loop completed, transitioning to speaking")
                    else:
                        logger.info("⚠️ Loop wait timeout, proceeding with transition")

                    # idle 스트림 중지
                    if connection.get("idle_task"):
                        idle_task = connection["idle_task"]
                        if idle_task and not idle_task.done():
                            logger.debug("Stopping idle stream for speaking")
                            idle_task.cancel()
                            try:
                                await asyncio.wait_for(idle_task, timeout=0.5)
                            except (asyncio.CancelledError, asyncio.TimeoutError):
                                pass
                            connection["idle_task"] = None

                    # SPEAKING 상태로 전이
                    state_machine.transition_to(ConnectionState.SPEAKING)
                    await self._send_status(websocket, "speaking")

                # 오디오 데이터 전송 (문장별로 전송 - 비디오 프레임 생성 완료 후)
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                await self._send_json(websocket, {
                    "type": "audio_data",
                    "data": audio_base64,
                    "sample_rate": self.pipeline.tts.sample_rate,
                    "frame_count": len(video_frames),
                    "sentence_index": idx,
                    "total_sentences": total,
                })

                logger.info(f"🔊 문장 {idx+1}/{total} 오디오 전송, 비디오 스트리밍 시작...")

                # 비디오 프레임을 FPS에 맞춰 전송 (오디오와 동기화)
                lipsync_frame_idx = 0
                for frame_data in video_frames:
                    frame_start = time.time()
                    lipsync_frame_idx += 1

                    if connection_id not in self._active_connections:
                        logger.warning("Connection closed during video streaming")
                        break

                    # 🔑 WebSocket 전송 직전 최종 프레임 크기 검증 및 강제 리사이즈
                    import cv2
                    import numpy as np

                    np_arr = np.frombuffer(frame_data, np.uint8)
                    decoded_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    if decoded_frame is not None:
                        actual_h, actual_w = decoded_frame.shape[:2]
                        target_w, target_h = 784, 1176

                        if actual_w != target_w or actual_h != target_h:
                            if lipsync_frame_idx <= 3 or lipsync_frame_idx % 30 == 0:
                                print(f"[LIPSYNC RESIZE] Frame #{lipsync_frame_idx}: {actual_w}x{actual_h} → {target_w}x{target_h}", flush=True)
                                logger.warning(f"⚠️ [LipSync] 프레임 크기 불일치: {actual_w}x{actual_h} → {target_w}x{target_h}로 강제 리사이즈")

                            resized_frame = cv2.resize(decoded_frame, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                            _, encoded = cv2.imencode(".jpg", resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            frame_data = encoded.tobytes()

                    try:
                        await websocket.send_bytes(frame_data)
                        total_frames_sent += 1
                    except RuntimeError as e:
                        if "close message" in str(e) or "disconnect" in str(e).lower():
                            logger.warning("WebSocket closed, stopping video stream")
                            break
                        raise

                    # FPS에 맞춰 대기
                    elapsed = time.time() - frame_start
                    sleep_time = frame_duration - elapsed
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)

                # 프레임 메모리 즉시 해제
                video_frames.clear()
                
                logger.info(f"✅ 문장 {idx+1}/{total} 완료")

            if not stream_started:
                logger.warning("⚠️ TTS 스트리밍이 시작되지 않았습니다. synthesize_sentences_streaming이 아무것도 yield하지 않았습니다.")
            
            logger.info(f"✅ 실시간 TTS 스트리밍 완료: 총 {total_frames_sent} 프레임 전송")

            # 스트리밍 완료 알림
            await self._send_json(websocket, {
                "type": "streaming_complete",
                "total_frames": total_frames_sent,
            })

            # 립싱크 완료 후 CONNECTED 상태로 전이 및 idle 스트림 재시작
            if connection_id in self._active_connections:
                state_machine.transition_to(ConnectionState.CONNECTED)
                await self._send_state_status(websocket, state_machine.state)

                # 서버 사이드 idle 스트림이 활성화된 경우에만 재시작
                disable_idle_stream = getattr(self.pipeline.settings, 'disable_server_idle_stream', True)
                if not disable_idle_stream:
                    connection = self._active_connections[connection_id]
                    if not connection.get("idle_task") or (hasattr(connection["idle_task"], 'done') and connection["idle_task"].done()):
                        logger.debug("Restarting idle stream after lip sync")
                        idle_task = asyncio.create_task(
                            self._start_idle_stream_background(websocket, session_id, connection_id)
                        )
                        connection["idle_task"] = idle_task
                else:
                    logger.debug("Server-side idle stream disabled, not restarting after lip sync")

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"❌ TTS/Lipsync streaming error: {e}", exc_info=True)
            logger.error(f"❌ 에러 상세:\n{error_trace}")
            state_machine.transition_to(ConnectionState.ERROR)
            # 클라이언트에 에러 알림 (상세 정보 포함)
            await self._send_json(websocket, {
                "type": "streaming_error",
                "error": str(e),
                "error_type": type(e).__name__,
                "message": f"TTS/립싱크 스트리밍 오류: {str(e)}",
            })

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