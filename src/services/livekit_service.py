"""
LiveKit Service for WebRTC streaming.

LiveKit 기반 WebRTC 미디어 스트리밍 서비스
특징:
- 양방향 실시간 미디어 스트리밍
- 룸 및 참가자 관리
- 토큰 생성
- Apache 2.0 라이선스
"""

import logging
from datetime import timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class LiveKitService:
    """
    LiveKit WebRTC 서비스

    Features:
    - 룸 생성 및 관리
    - 참가자 토큰 생성
    - 미디어 트랙 관리
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        url: str,
    ):
        """
        Initialize LiveKit Service.

        Args:
            api_key: LiveKit API key
            api_secret: LiveKit API secret
            url: LiveKit server URL
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.url = url

        self._api = None
        self._access_token = None
        self._initialized = False

    async def initialize(self) -> None:
        """서비스 초기화"""
        if self._initialized:
            return

        logger.info("Initializing LiveKit Service...")

        try:
            from livekit import api
            from livekit.api import access_token

            self._api = api
            self._access_token = access_token
            self._initialized = True
            logger.info("LiveKit Service initialized")

        except ImportError:
            logger.warning(
                "LiveKit SDK not installed. WebRTC features will be disabled. "
                "Install with: pip install livekit livekit-api"
            )

    def create_token(
        self,
        room_name: str,
        participant_name: str,
        participant_identity: Optional[str] = None,
        ttl: timedelta = timedelta(hours=24),
        can_publish: bool = True,
        can_subscribe: bool = True,
        can_publish_data: bool = True,
    ) -> str:
        """
        참가자 토큰 생성

        Args:
            room_name: 룸 이름
            participant_name: 참가자 표시 이름
            participant_identity: 참가자 고유 ID (없으면 이름 사용)
            ttl: 토큰 유효 시간
            can_publish: 미디어 발행 권한
            can_subscribe: 미디어 구독 권한
            can_publish_data: 데이터 채널 발행 권한

        Returns:
            JWT 토큰 문자열
        """
        if not self._initialized or self._api is None:
            logger.warning("LiveKit not initialized, returning empty token")
            return ""

        identity = participant_identity or participant_name

        # AccessToken 생성 (access_token 모듈 사용)
        token = self._access_token.AccessToken(
            api_key=self.api_key,
            api_secret=self.api_secret,
        )

        # 권한 설정
        token.with_identity(identity)
        token.with_name(participant_name)
        token.with_ttl(ttl)

        # VideoGrants 설정 (VideoGrant가 아닌 VideoGrants 사용)
        grant = self._access_token.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=can_publish,
            can_subscribe=can_subscribe,
            can_publish_data=can_publish_data,
        )
        token.with_grants(grant)

        return token.to_jwt()

    async def create_room(
        self,
        room_name: str,
        empty_timeout: int = 300,
        max_participants: int = 10,
    ) -> Optional[dict]:
        """
        룸 생성

        Args:
            room_name: 룸 이름
            empty_timeout: 빈 룸 타임아웃 (초)
            max_participants: 최대 참가자 수

        Returns:
            생성된 룸 정보 또는 None
        """
        if not self._initialized or self._api is None:
            return None

        try:
            room_service = self._api.RoomServiceClient(
                self.url,
                api_key=self.api_key,
                api_secret=self.api_secret,
            )

            room = await room_service.create_room(
                self._api.CreateRoomRequest(
                    name=room_name,
                    empty_timeout=empty_timeout,
                    max_participants=max_participants,
                )
            )

            return {
                "name": room.name,
                "sid": room.sid,
                "creation_time": room.creation_time,
                "num_participants": room.num_participants,
            }

        except Exception as e:
            logger.error(f"Failed to create room: {e}")
            return None

    async def delete_room(self, room_name: str) -> bool:
        """
        룸 삭제

        Args:
            room_name: 삭제할 룸 이름

        Returns:
            성공 여부
        """
        if not self._initialized or self._api is None:
            return False

        try:
            room_service = self._api.RoomServiceClient(
                self.url,
                api_key=self.api_key,
                api_secret=self.api_secret,
            )

            await room_service.delete_room(
                self._api.DeleteRoomRequest(room=room_name)
            )
            return True

        except Exception as e:
            logger.error(f"Failed to delete room: {e}")
            return False

    async def list_rooms(self) -> list:
        """
        모든 룸 목록 조회

        Returns:
            룸 정보 리스트
        """
        if not self._initialized or self._api is None:
            return []

        try:
            room_service = self._api.RoomServiceClient(
                self.url,
                api_key=self.api_key,
                api_secret=self.api_secret,
            )

            response = await room_service.list_rooms(
                self._api.ListRoomsRequest()
            )

            return [
                {
                    "name": room.name,
                    "sid": room.sid,
                    "num_participants": room.num_participants,
                }
                for room in response.rooms
            ]

        except Exception as e:
            logger.error(f"Failed to list rooms: {e}")
            return []

    async def list_participants(self, room_name: str) -> list:
        """
        룸 내 참가자 목록 조회

        Args:
            room_name: 룸 이름

        Returns:
            참가자 정보 리스트
        """
        if not self._initialized or self._api is None:
            return []

        try:
            room_service = self._api.RoomServiceClient(
                self.url,
                api_key=self.api_key,
                api_secret=self.api_secret,
            )

            response = await room_service.list_participants(
                self._api.ListParticipantsRequest(room=room_name)
            )

            return [
                {
                    "identity": p.identity,
                    "name": p.name,
                    "state": p.state,
                    "joined_at": p.joined_at,
                }
                for p in response.participants
            ]

        except Exception as e:
            logger.error(f"Failed to list participants: {e}")
            return []

    async def remove_participant(
        self, room_name: str, identity: str
    ) -> bool:
        """
        룸에서 참가자 제거

        Args:
            room_name: 룸 이름
            identity: 참가자 ID

        Returns:
            성공 여부
        """
        if not self._initialized or self._api is None:
            return False

        try:
            room_service = self._api.RoomServiceClient(
                self.url,
                api_key=self.api_key,
                api_secret=self.api_secret,
            )

            await room_service.remove_participant(
                self._api.RoomParticipantIdentity(
                    room=room_name,
                    identity=identity,
                )
            )
            return True

        except Exception as e:
            logger.error(f"Failed to remove participant: {e}")
            return False

    def get_websocket_url(self) -> str:
        """WebSocket URL 반환"""
        return self.url

    async def cleanup(self) -> None:
        """리소스 정리"""
        self._api = None
        self._access_token = None
        self._initialized = False
        logger.info("LiveKit Service cleaned up")
