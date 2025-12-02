"""
Model integrations package.

실제 ML 모델 통합 모듈
- MuseTalk: 립싱크
- LivePortrait: 얼굴 애니메이션
- Chatterbox: TTS (영어)
- Zonos: TTS with Voice Cloning (다국어, 한국어 미지원)
"""

from .musetalk import MuseTalkModel
from .live_portrait import LivePortraitModel
from .chatterbox_tts import ChatterboxTTSModel
from .zonos_tts import ZonosTTSModel

__all__ = [
    "MuseTalkModel",
    "LivePortraitModel",
    "ChatterboxTTSModel",
    "ZonosTTSModel",
]
