"""
Model integrations package.

실제 ML 모델 통합 모듈
- MuseTalk: 립싱크
- LivePortrait: 얼굴 애니메이션
- Chatterbox: TTS
"""

from .musetalk import MuseTalkModel
from .live_portrait import LivePortraitModel
from .chatterbox_tts import ChatterboxTTSModel

__all__ = [
    "MuseTalkModel",
    "LivePortraitModel",
    "ChatterboxTTSModel",
]
