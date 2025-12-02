#!/usr/bin/env python3
"""
Zonos TTS Voice Manager - Standalone Server

독립 실행 가능한 Zonos TTS 음성 관리 서버
- 웹 기반 음성 관리 인터페이스
- 음성 복제 (Voice Cloning)
- 음성 미리듣기 및 다운로드
- REST API 제공
"""

import argparse
import asyncio
import io
import json
import logging
import os
import sys
import uuid
import wave
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, Form
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, Response, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    logger.error("FastAPI가 설치되지 않았습니다. pip install fastapi uvicorn python-multipart")
    sys.exit(1)

# NumPy
try:
    import numpy as np
except ImportError:
    logger.error("NumPy가 설치되지 않았습니다. pip install numpy")
    sys.exit(1)

# ============ 데이터 클래스 ============

@dataclass
class VoiceProfile:
    """음성 프로필"""
    id: str
    name: str
    description: str
    language: str
    audio_path: str
    embedding_path: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    duration_seconds: float = 0.0
    sample_rate: int = 44100

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at


# ============ Zonos TTS 래퍼 ============

class ZonosTTSManager:
    """Zonos TTS 관리자"""

    SUPPORTED_LANGUAGES = ["ko", "en", "ja", "zh", "fr", "de"]
    EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"]

    def __init__(self, voices_dir: str = "assets/voices", device: str = "cuda"):
        self.voices_dir = Path(voices_dir)
        self.device = device
        self.sample_rate = 44100

        self._model = None
        self._initialized = False
        self._voice_profiles: Dict[str, VoiceProfile] = {}
        self._speaker_embeddings: Dict[str, Any] = {}

        # 디렉토리 생성
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        (self.voices_dir / "audio").mkdir(exist_ok=True)
        (self.voices_dir / "embeddings").mkdir(exist_ok=True)

    async def initialize(self) -> bool:
        """초기화"""
        if self._initialized:
            return True

        try:
            # Zonos 모델 로드 시도
            try:
                import torch
                from zonos.model import Zonos

                self._model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=self.device)
                logger.info(f"Zonos TTS 모델 로드 성공 (device: {self.device})")
            except ImportError:
                logger.warning("Zonos 패키지가 설치되지 않았습니다. Mock 모드로 실행됩니다.")
                self._model = None
            except Exception as e:
                logger.warning(f"Zonos 모델 로드 실패: {e}. Mock 모드로 실행됩니다.")
                self._model = None

            # 저장된 프로필 로드
            await self._load_profiles()

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"초기화 실패: {e}")
            self._initialized = True
            return False

    async def _load_profiles(self):
        """저장된 프로필 로드"""
        profiles_file = self.voices_dir / "profiles.json"

        if profiles_file.exists():
            try:
                with open(profiles_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for profile_data in data.get("profiles", []):
                    profile = VoiceProfile(**profile_data)
                    self._voice_profiles[profile.id] = profile

                logger.info(f"{len(self._voice_profiles)}개 음성 프로필 로드됨")
            except Exception as e:
                logger.error(f"프로필 로드 실패: {e}")

    async def _save_profiles(self):
        """프로필 저장"""
        profiles_file = self.voices_dir / "profiles.json"

        try:
            data = {
                "profiles": [asdict(p) for p in self._voice_profiles.values()],
                "updated_at": datetime.now().isoformat(),
            }
            with open(profiles_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"프로필 저장 실패: {e}")

    async def create_voice(
        self,
        name: str,
        audio_data: bytes,
        description: str = "",
        language: str = "ko",
    ) -> Optional[VoiceProfile]:
        """새 음성 프로필 생성"""
        voice_id = str(uuid.uuid4())[:8]

        try:
            # 오디오 파일 저장
            audio_path = self.voices_dir / "audio" / f"{voice_id}.wav"
            duration = await self._save_audio(audio_data, audio_path)

            # 스피커 임베딩 생성 (Zonos가 있을 때만)
            embedding_path = None
            if self._model is not None:
                try:
                    import torch
                    import torchaudio

                    wav, sr = torchaudio.load(str(audio_path))
                    speaker_embedding = self._model.make_speaker_embedding(wav, sr)

                    embedding_path = str(self.voices_dir / "embeddings" / f"{voice_id}.pt")
                    torch.save(speaker_embedding, embedding_path)
                    self._speaker_embeddings[voice_id] = speaker_embedding

                    logger.info(f"스피커 임베딩 생성: {voice_id}")
                except Exception as e:
                    logger.warning(f"임베딩 생성 실패: {e}")

            # 프로필 생성
            profile = VoiceProfile(
                id=voice_id,
                name=name,
                description=description,
                language=language,
                audio_path=str(audio_path),
                embedding_path=embedding_path,
                duration_seconds=duration,
                sample_rate=self.sample_rate,
            )

            self._voice_profiles[voice_id] = profile
            await self._save_profiles()

            logger.info(f"음성 프로필 생성: {voice_id} ({name})")
            return profile

        except Exception as e:
            logger.error(f"음성 생성 실패: {e}")
            return None

    async def _save_audio(self, audio_data: bytes, output_path: Path) -> float:
        """오디오 저장 및 길이 반환"""
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_file(io.BytesIO(audio_data))
            audio = audio.set_frame_rate(self.sample_rate).set_channels(1)
            audio.export(str(output_path), format="wav")

            return len(audio) / 1000.0
        except ImportError:
            # pydub 없으면 직접 저장
            with open(output_path, "wb") as f:
                f.write(audio_data)
            return 0.0

    def list_voices(self) -> List[VoiceProfile]:
        """모든 음성 목록"""
        return list(self._voice_profiles.values())

    def get_voice(self, voice_id: str) -> Optional[VoiceProfile]:
        """음성 조회"""
        return self._voice_profiles.get(voice_id)

    async def update_voice(
        self,
        voice_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Optional[VoiceProfile]:
        """음성 수정"""
        if voice_id not in self._voice_profiles:
            return None

        profile = self._voice_profiles[voice_id]
        if name:
            profile.name = name
        if description is not None:
            profile.description = description
        if language and language in self.SUPPORTED_LANGUAGES:
            profile.language = language

        profile.updated_at = datetime.now().isoformat()
        await self._save_profiles()

        return profile

    async def delete_voice(self, voice_id: str) -> bool:
        """음성 삭제"""
        if voice_id not in self._voice_profiles:
            return False

        profile = self._voice_profiles[voice_id]

        try:
            if profile.audio_path and Path(profile.audio_path).exists():
                os.remove(profile.audio_path)
            if profile.embedding_path and Path(profile.embedding_path).exists():
                os.remove(profile.embedding_path)

            del self._voice_profiles[voice_id]
            if voice_id in self._speaker_embeddings:
                del self._speaker_embeddings[voice_id]

            await self._save_profiles()
            return True
        except Exception as e:
            logger.error(f"삭제 실패: {e}")
            return False

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "ko",
        emotion: str = "neutral",
        speaking_rate: float = 1.0,
    ) -> np.ndarray:
        """TTS 합성"""
        if not text.strip():
            return np.array([], dtype=np.float32)

        if self._model is None:
            return self._generate_mock_audio(len(text))

        try:
            import torch

            speaker_embedding = None
            if voice_id and voice_id in self._speaker_embeddings:
                speaker_embedding = self._speaker_embeddings[voice_id]

            cond_dict = self._model.make_cond_dict(
                text=text,
                speaker=speaker_embedding,
                language=language,
                emotion=[0.0] * 8,
            )

            cond_dict["speaking_rate"] = torch.tensor([speaking_rate], device=self.device)

            with torch.no_grad():
                audio = self._model.generate(cond_dict)

            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()

            if audio.ndim > 1:
                audio = audio.squeeze()

            audio = audio.astype(np.float32)
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()

            return audio

        except Exception as e:
            logger.error(f"합성 실패: {e}")
            return self._generate_mock_audio(len(text))

    def _generate_mock_audio(self, text_length: int) -> np.ndarray:
        """Mock 오디오 생성"""
        duration_samples = max(int(text_length * 0.1 * self.sample_rate), self.sample_rate // 2)
        t = np.linspace(0, duration_samples / self.sample_rate, duration_samples)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return audio

    async def get_voice_audio(self, voice_id: str) -> Optional[bytes]:
        """원본 오디오 반환"""
        profile = self.get_voice(voice_id)
        if not profile or not profile.audio_path:
            return None

        try:
            with open(profile.audio_path, "rb") as f:
                return f.read()
        except:
            return None


# ============ FastAPI 앱 ============

app = FastAPI(
    title="Zonos TTS Voice Manager",
    description="음성 복제 및 관리 시스템",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 매니저
manager: Optional[ZonosTTSManager] = None


# ============ HTML 템플릿 ============

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zonos TTS Voice Manager</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background: #0f172a; color: white; }
        .card { background: #1e293b; border-radius: 12px; }
    </style>
</head>
<body class="min-h-screen p-4">
    <div class="max-w-4xl mx-auto">
        <header class="text-center mb-8">
            <h1 class="text-3xl font-bold mb-2">🎤 Zonos TTS Voice Manager</h1>
            <p class="text-gray-400">음성 복제 및 관리 시스템 (한국어 지원)</p>
        </header>

        <!-- 새 음성 생성 -->
        <div class="card p-6 mb-6">
            <h2 class="text-xl font-semibold mb-4">➕ 새 음성 복제</h2>
            <form id="createForm" class="space-y-4">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-sm text-gray-400 mb-1">이름 *</label>
                        <input type="text" id="voiceName" required
                            class="w-full px-3 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none">
                    </div>
                    <div>
                        <label class="block text-sm text-gray-400 mb-1">언어</label>
                        <select id="voiceLang" class="w-full px-3 py-2 bg-gray-700 rounded-lg border border-gray-600">
                            <option value="ko">한국어</option>
                            <option value="en">English</option>
                            <option value="ja">日本語</option>
                            <option value="zh">中文</option>
                            <option value="fr">Français</option>
                            <option value="de">Deutsch</option>
                        </select>
                    </div>
                </div>
                <div>
                    <label class="block text-sm text-gray-400 mb-1">설명</label>
                    <input type="text" id="voiceDesc"
                        class="w-full px-3 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
                        placeholder="선택사항">
                </div>
                <div>
                    <label class="block text-sm text-gray-400 mb-1">참조 오디오 * (10-30초 권장)</label>
                    <input type="file" id="audioFile" accept="audio/*" required
                        class="w-full px-3 py-2 bg-gray-700 rounded-lg border border-gray-600 file:mr-4 file:py-1 file:px-3 file:rounded file:border-0 file:bg-blue-600 file:text-white file:cursor-pointer">
                </div>
                <button type="submit" id="createBtn"
                    class="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition-colors">
                    🎙️ 음성 복제 시작
                </button>
            </form>
        </div>

        <!-- 미리듣기 설정 -->
        <div class="card p-6 mb-6">
            <h2 class="text-xl font-semibold mb-4">🔊 미리듣기 설정</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="md:col-span-2">
                    <label class="block text-sm text-gray-400 mb-1">텍스트</label>
                    <input type="text" id="previewText" value="안녕하세요, 반갑습니다."
                        class="w-full px-3 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none">
                </div>
                <div>
                    <label class="block text-sm text-gray-400 mb-1">속도: <span id="rateVal">1.0</span>x</label>
                    <input type="range" id="speakingRate" min="0.5" max="2" step="0.1" value="1.0"
                        class="w-full" oninput="document.getElementById('rateVal').textContent=this.value">
                </div>
            </div>
        </div>

        <!-- 음성 목록 -->
        <div class="card p-6">
            <div class="flex justify-between items-center mb-4">
                <h2 class="text-xl font-semibold">📋 음성 목록</h2>
                <button onclick="loadVoices()" class="px-3 py-1 bg-gray-600 hover:bg-gray-500 rounded-lg text-sm">
                    🔄 새로고침
                </button>
            </div>
            <div id="voiceList" class="space-y-3">
                <p class="text-gray-400 text-center py-4">로딩 중...</p>
            </div>
        </div>

        <footer class="text-center mt-8 text-gray-500 text-sm">
            <p>Zonos TTS Voice Manager v1.0 | 한국어 지원 (비공식)</p>
        </footer>
    </div>

    <audio id="audioPlayer" class="hidden"></audio>

    <script>
        const API = '';

        async function loadVoices() {
            try {
                const res = await fetch(`${API}/api/voices`);
                const data = await res.json();

                const list = document.getElementById('voiceList');
                if (data.voices.length === 0) {
                    list.innerHTML = '<p class="text-gray-400 text-center py-4">등록된 음성이 없습니다. 위에서 새 음성을 복제해 주세요.</p>';
                    return;
                }

                list.innerHTML = data.voices.map(v => `
                    <div class="flex items-center justify-between p-4 bg-gray-800 rounded-lg">
                        <div class="flex-1">
                            <div class="flex items-center gap-2">
                                <span class="font-semibold">${v.name}</span>
                                <span class="px-2 py-0.5 bg-gray-700 rounded text-xs">${getLangName(v.language)}</span>
                                <span class="text-xs text-gray-500">${formatDuration(v.duration_seconds)}</span>
                            </div>
                            ${v.description ? `<p class="text-sm text-gray-400 mt-1">${v.description}</p>` : ''}
                            <p class="text-xs text-gray-500 mt-1">ID: ${v.id}</p>
                        </div>
                        <div class="flex gap-2 ml-4">
                            <button onclick="previewVoice('${v.id}')" class="p-2 bg-green-600 hover:bg-green-700 rounded-lg" title="미리듣기">▶️</button>
                            <button onclick="downloadAudio('${v.id}', '${v.name}')" class="p-2 bg-gray-600 hover:bg-gray-500 rounded-lg" title="다운로드">⬇️</button>
                            <button onclick="deleteVoice('${v.id}')" class="p-2 bg-red-600 hover:bg-red-700 rounded-lg" title="삭제">🗑️</button>
                        </div>
                    </div>
                `).join('');
            } catch (e) {
                console.error(e);
                document.getElementById('voiceList').innerHTML = '<p class="text-red-400 text-center py-4">로드 실패</p>';
            }
        }

        function getLangName(code) {
            const names = {ko: '한국어', en: 'English', ja: '日本語', zh: '中文', fr: 'Français', de: 'Deutsch'};
            return names[code] || code;
        }

        function formatDuration(sec) {
            if (!sec) return '';
            const m = Math.floor(sec / 60);
            const s = Math.floor(sec % 60);
            return m > 0 ? `${m}:${s.toString().padStart(2, '0')}` : `${s}초`;
        }

        document.getElementById('createForm').onsubmit = async (e) => {
            e.preventDefault();
            const btn = document.getElementById('createBtn');
            btn.disabled = true;
            btn.textContent = '⏳ 처리 중...';

            try {
                const formData = new FormData();
                formData.append('audio', document.getElementById('audioFile').files[0]);
                formData.append('name', document.getElementById('voiceName').value);
                formData.append('description', document.getElementById('voiceDesc').value);
                formData.append('language', document.getElementById('voiceLang').value);

                const res = await fetch(`${API}/api/voices`, { method: 'POST', body: formData });
                if (!res.ok) throw new Error('생성 실패');

                alert('✅ 음성이 생성되었습니다!');
                document.getElementById('createForm').reset();
                loadVoices();
            } catch (e) {
                alert('❌ 오류: ' + e.message);
            } finally {
                btn.disabled = false;
                btn.textContent = '🎙️ 음성 복제 시작';
            }
        };

        async function previewVoice(voiceId) {
            try {
                const text = document.getElementById('previewText').value;
                const rate = document.getElementById('speakingRate').value;

                const res = await fetch(`${API}/api/voices/${voiceId}/preview`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text, voice_id: voiceId, speaking_rate: parseFloat(rate)})
                });

                if (!res.ok) throw new Error('미리듣기 실패');

                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                const player = document.getElementById('audioPlayer');
                player.src = url;
                player.play();
            } catch (e) {
                alert('❌ 오류: ' + e.message);
            }
        }

        async function downloadAudio(voiceId, name) {
            try {
                const res = await fetch(`${API}/api/voices/${voiceId}/audio`);
                if (!res.ok) throw new Error('다운로드 실패');

                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${name}.wav`;
                a.click();
            } catch (e) {
                alert('❌ 오류: ' + e.message);
            }
        }

        async function deleteVoice(voiceId) {
            if (!confirm('정말 삭제하시겠습니까?')) return;

            try {
                const res = await fetch(`${API}/api/voices/${voiceId}`, {method: 'DELETE'});
                if (!res.ok) throw new Error('삭제 실패');
                loadVoices();
            } catch (e) {
                alert('❌ 오류: ' + e.message);
            }
        }

        // 초기 로드
        loadVoices();
    </script>
</body>
</html>'''


# ============ API 라우트 ============

@app.on_event("startup")
async def startup():
    global manager
    manager = ZonosTTSManager()
    await manager.initialize()


@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 페이지"""
    return HTML_TEMPLATE


@app.get("/api/voices")
async def list_voices():
    """음성 목록"""
    voices = manager.list_voices()
    return {
        "voices": [asdict(v) for v in voices],
        "total": len(voices),
        "supported_languages": ZonosTTSManager.SUPPORTED_LANGUAGES,
    }


@app.post("/api/voices")
async def create_voice(
    audio: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(""),
    language: str = Form("ko"),
):
    """새 음성 생성"""
    content = await audio.read()

    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(400, "파일이 너무 큽니다 (최대 50MB)")

    profile = await manager.create_voice(
        name=name,
        audio_data=content,
        description=description,
        language=language,
    )

    if not profile:
        raise HTTPException(500, "음성 생성 실패")

    return asdict(profile)


@app.get("/api/voices/{voice_id}")
async def get_voice(voice_id: str):
    """음성 조회"""
    profile = manager.get_voice(voice_id)
    if not profile:
        raise HTTPException(404, "음성을 찾을 수 없습니다")
    return asdict(profile)


@app.delete("/api/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """음성 삭제"""
    success = await manager.delete_voice(voice_id)
    if not success:
        raise HTTPException(404, "음성을 찾을 수 없습니다")
    return {"status": "deleted"}


@app.get("/api/voices/{voice_id}/audio")
async def get_voice_audio(voice_id: str):
    """원본 오디오 다운로드"""
    audio_data = await manager.get_voice_audio(voice_id)
    if not audio_data:
        raise HTTPException(404, "오디오를 찾을 수 없습니다")

    return Response(
        content=audio_data,
        media_type="audio/wav",
        headers={"Content-Disposition": f"attachment; filename={voice_id}.wav"},
    )


@app.post("/api/voices/{voice_id}/preview")
async def preview_voice(voice_id: str, request: dict):
    """음성 미리듣기"""
    text = request.get("text", "안녕하세요")
    speaking_rate = request.get("speaking_rate", 1.0)

    audio = await manager.synthesize(
        text=text,
        voice_id=voice_id,
        language=manager.get_voice(voice_id).language if manager.get_voice(voice_id) else "ko",
        speaking_rate=speaking_rate,
    )

    if len(audio) == 0:
        raise HTTPException(500, "합성 실패")

    # WAV 변환
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(manager.sample_rate)
        wav_file.writeframes((audio * 32767).astype(np.int16).tobytes())

    wav_buffer.seek(0)
    return Response(content=wav_buffer.read(), media_type="audio/wav")


# ============ 메인 ============

def main():
    parser = argparse.ArgumentParser(description="Zonos TTS Voice Manager")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8100, help="Port (default: 8100)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    print(f"\n🎤 Zonos TTS Voice Manager 시작")
    print(f"   http://{args.host}:{args.port}/\n")

    uvicorn.run(
        "scripts.zonos_server:app" if __name__ != "__main__" else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
