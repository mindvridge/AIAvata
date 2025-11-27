"""
Emotion definitions and mappings.

감정 enum 및 감정 매핑 정의
"""

from enum import Enum
from typing import Dict


class Emotion(str, Enum):
    """사용자 및 아바타 감정 상태"""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"

    # 아바타 전용 상태
    LISTENING = "listening"
    THINKING = "thinking"
    SYMPATHETIC = "sympathetic"
    CONCERNED = "concerned"


class EmotionMapping:
    """
    감정 매핑 유틸리티

    - SenseVoice 감정 ID → Emotion enum
    - 사용자 감정 → 아바타 반응 감정
    """

    # SenseVoice 감정 ID 매핑
    SENSEVOICE_MAP: Dict[int, Emotion] = {
        0: Emotion.ANGRY,
        1: Emotion.HAPPY,
        2: Emotion.NEUTRAL,
        3: Emotion.SAD,
        4: Emotion.NEUTRAL,  # unknown → neutral
    }

    # 사용자 감정 → 아바타 반응 매핑
    USER_TO_AVATAR_MAP: Dict[Emotion, Emotion] = {
        Emotion.HAPPY: Emotion.HAPPY,  # 사용자가 기쁘면 아바타도 기쁨
        Emotion.SAD: Emotion.SYMPATHETIC,  # 사용자가 슬프면 아바타는 공감
        Emotion.ANGRY: Emotion.CONCERNED,  # 사용자가 화나면 아바타는 걱정
        Emotion.SURPRISED: Emotion.NEUTRAL,  # 사용자가 놀라면 아바타는 중립
        Emotion.FEARFUL: Emotion.SYMPATHETIC,  # 사용자가 두려우면 아바타는 공감
        Emotion.DISGUSTED: Emotion.CONCERNED,  # 사용자가 혐오스러우면 아바타는 걱정
        Emotion.NEUTRAL: Emotion.NEUTRAL,  # 중립은 중립
    }

    # Idle 루프 파일명 매핑
    IDLE_LOOP_FILES: Dict[Emotion, str] = {
        Emotion.NEUTRAL: "neutral_idle.mp4",
        Emotion.HAPPY: "happy_smile.mp4",
        Emotion.SAD: "sad_idle.mp4",
        Emotion.SYMPATHETIC: "sympathetic.mp4",
        Emotion.CONCERNED: "concerned.mp4",
        Emotion.LISTENING: "listening_nod.mp4",
        Emotion.THINKING: "thinking.mp4",
    }

    @classmethod
    def from_sensevoice(cls, emotion_id: int) -> Emotion:
        """SenseVoice 감정 ID를 Emotion enum으로 변환"""
        return cls.SENSEVOICE_MAP.get(emotion_id, Emotion.NEUTRAL)

    @classmethod
    def get_avatar_response(cls, user_emotion: Emotion) -> Emotion:
        """사용자 감정에 대한 아바타 반응 감정 반환"""
        return cls.USER_TO_AVATAR_MAP.get(user_emotion, Emotion.NEUTRAL)

    @classmethod
    def get_idle_loop_filename(cls, emotion: Emotion) -> str:
        """감정에 해당하는 idle 루프 파일명 반환"""
        return cls.IDLE_LOOP_FILES.get(emotion, "neutral_idle.mp4")
