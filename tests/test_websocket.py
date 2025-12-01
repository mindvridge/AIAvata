"""
Tests for WebSocket State Machine and Connection Handling.
"""

import pytest
import time

from src.models.schemas import ConnectionState, CONNECTION_STATE_TRANSITIONS
from src.api.websocket import ConnectionStateMachine


class TestConnectionStateMachine:
    """ConnectionStateMachine 테스트"""

    def test_initial_state(self):
        """초기 상태 테스트"""
        sm = ConnectionStateMachine()
        assert sm.state == ConnectionState.CONNECTING

    def test_initial_state_custom(self):
        """커스텀 초기 상태 테스트"""
        sm = ConnectionStateMachine(initial_state=ConnectionState.CONNECTED)
        assert sm.state == ConnectionState.CONNECTED

    def test_valid_transition(self):
        """유효한 상태 전이 테스트"""
        sm = ConnectionStateMachine()

        # CONNECTING -> CONNECTED
        assert sm.transition_to(ConnectionState.CONNECTED) is True
        assert sm.state == ConnectionState.CONNECTED

        # CONNECTED -> IDLE_STREAMING
        assert sm.transition_to(ConnectionState.IDLE_STREAMING) is True
        assert sm.state == ConnectionState.IDLE_STREAMING

        # IDLE_STREAMING -> PROCESSING
        assert sm.transition_to(ConnectionState.PROCESSING) is True
        assert sm.state == ConnectionState.PROCESSING

        # PROCESSING -> SPEAKING
        assert sm.transition_to(ConnectionState.SPEAKING) is True
        assert sm.state == ConnectionState.SPEAKING

    def test_invalid_transition(self):
        """유효하지 않은 상태 전이 테스트"""
        sm = ConnectionStateMachine()

        # CONNECTING -> SPEAKING (직접 불가)
        assert sm.transition_to(ConnectionState.SPEAKING) is False
        assert sm.state == ConnectionState.CONNECTING  # 상태 유지

    def test_force_transition(self):
        """강제 전이 테스트"""
        sm = ConnectionStateMachine()

        # force=True로 유효하지 않은 전이도 허용
        assert sm.transition_to(ConnectionState.SPEAKING, force=True) is True
        assert sm.state == ConnectionState.SPEAKING

    def test_can_transition_to(self):
        """전이 가능 여부 확인 테스트"""
        sm = ConnectionStateMachine(initial_state=ConnectionState.CONNECTED)

        assert sm.can_transition_to(ConnectionState.IDLE_STREAMING) is True
        assert sm.can_transition_to(ConnectionState.PROCESSING) is True
        assert sm.can_transition_to(ConnectionState.RECONNECTING) is False

    def test_is_active(self):
        """활성 상태 확인 테스트"""
        sm = ConnectionStateMachine(initial_state=ConnectionState.CONNECTED)
        assert sm.is_active() is True

        sm.transition_to(ConnectionState.IDLE_STREAMING)
        assert sm.is_active() is True

        sm.transition_to(ConnectionState.DISCONNECTED, force=True)
        assert sm.is_active() is False

    def test_is_busy(self):
        """처리 중 상태 확인 테스트"""
        sm = ConnectionStateMachine(initial_state=ConnectionState.CONNECTED)
        assert sm.is_busy() is False

        sm.transition_to(ConnectionState.PROCESSING)
        assert sm.is_busy() is True

        sm.transition_to(ConnectionState.SPEAKING)
        assert sm.is_busy() is True

        sm.transition_to(ConnectionState.CONNECTED)
        assert sm.is_busy() is False

    def test_state_history(self):
        """상태 이력 테스트"""
        sm = ConnectionStateMachine()

        sm.transition_to(ConnectionState.CONNECTED)
        sm.transition_to(ConnectionState.IDLE_STREAMING)

        history = sm.state_history
        assert len(history) == 3
        assert history[0][1] == ConnectionState.CONNECTING
        assert history[1][1] == ConnectionState.CONNECTED
        assert history[2][1] == ConnectionState.IDLE_STREAMING

    def test_state_history_limit(self):
        """상태 이력 크기 제한 테스트"""
        sm = ConnectionStateMachine()

        # 100개 이상의 전이
        for _ in range(150):
            sm.transition_to(ConnectionState.CONNECTED, force=True)
            sm.transition_to(ConnectionState.IDLE_STREAMING, force=True)

        # 최근 100개만 유지
        assert len(sm.state_history) <= 100

    def test_callback_on_state_change(self):
        """상태 변경 콜백 테스트"""
        callback_calls = []

        def on_change(old_state, new_state):
            callback_calls.append((old_state, new_state))

        sm = ConnectionStateMachine(on_state_change=on_change)
        sm.transition_to(ConnectionState.CONNECTED)

        assert len(callback_calls) == 1
        assert callback_calls[0] == (ConnectionState.CONNECTING, ConnectionState.CONNECTED)


class TestConnectionStateTransitions:
    """상태 전이 정의 테스트"""

    def test_all_states_defined(self):
        """모든 상태가 전이 맵에 정의되어 있는지 확인"""
        for state in ConnectionState:
            assert state in CONNECTION_STATE_TRANSITIONS

    def test_disconnected_can_reconnect(self):
        """연결 해제 상태에서 재연결 가능 확인"""
        transitions = CONNECTION_STATE_TRANSITIONS[ConnectionState.DISCONNECTED]
        assert ConnectionState.RECONNECTING in transitions
        assert ConnectionState.CONNECTING in transitions

    def test_error_recovery(self):
        """에러 상태에서 복구 가능 확인"""
        transitions = CONNECTION_STATE_TRANSITIONS[ConnectionState.ERROR]
        assert ConnectionState.CONNECTED in transitions
        assert ConnectionState.DISCONNECTED in transitions
