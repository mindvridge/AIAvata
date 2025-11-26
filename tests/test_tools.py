"""
Tests for Tools.

아바타 생성, 음성 샘플 준비, idle 루프 생성 도구 테스트
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add tools directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tools"))


class TestCreateSampleAvatar:
    """샘플 아바타 생성 테스트"""

    def test_create_gradient_background(self):
        """그라데이션 배경 생성 테스트"""
        from tools.create_sample_avatar import create_gradient_background

        bg = create_gradient_background(512, 512)

        assert isinstance(bg, np.ndarray)
        assert bg.shape == (512, 512, 3)
        assert bg.dtype == np.uint8

    def test_create_avatar_image(self):
        """아바타 이미지 생성 테스트"""
        from tools.create_sample_avatar import create_avatar_image

        avatar = create_avatar_image(512, 512, "smile")

        assert isinstance(avatar, np.ndarray)
        assert avatar.shape == (512, 512, 3)
        assert avatar.dtype == np.uint8

    def test_create_avatar_image_neutral(self):
        """중립 표정 아바타 이미지 생성 테스트"""
        from tools.create_sample_avatar import create_avatar_image

        avatar = create_avatar_image(512, 512, "neutral")

        assert isinstance(avatar, np.ndarray)
        assert avatar.shape == (512, 512, 3)

    def test_create_avatar_different_sizes(self):
        """다양한 크기의 아바타 이미지 생성 테스트"""
        from tools.create_sample_avatar import create_avatar_image

        sizes = [256, 512, 1024]

        for size in sizes:
            avatar = create_avatar_image(size, size)
            assert avatar.shape == (size, size, 3), f"Failed for size {size}"


class TestPrepareVoiceSample:
    """음성 샘플 준비 테스트"""

    @pytest.fixture
    def sample_audio(self):
        """테스트 오디오 생성"""
        duration = 1.0
        sample_rate = 24000
        samples = int(duration * sample_rate)
        t = np.linspace(0, duration, samples)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        return audio.astype(np.float32)

    def test_normalize_audio(self, sample_audio):
        """오디오 정규화 테스트"""
        from tools.prepare_voice_sample import normalize_audio

        # 매우 큰 오디오
        loud_audio = sample_audio * 10

        normalized = normalize_audio(loud_audio, target_db=-20.0)

        assert isinstance(normalized, np.ndarray)
        assert np.abs(normalized).max() <= 1.0

    def test_remove_silence(self, sample_audio):
        """무음 제거 테스트"""
        from tools.prepare_voice_sample import remove_silence

        # 앞뒤에 무음 추가
        silence = np.zeros(5000, dtype=np.float32)
        audio_with_silence = np.concatenate([silence, sample_audio, silence])

        trimmed = remove_silence(audio_with_silence, sample_rate=24000)

        # 무음이 제거되어 길이가 짧아져야 함
        assert len(trimmed) < len(audio_with_silence)

    def test_validate_audio(self, sample_audio):
        """오디오 검증 테스트"""
        from tools.prepare_voice_sample import validate_audio

        result = validate_audio(sample_audio, sample_rate=24000)

        assert "duration" in result
        assert "sample_rate" in result
        assert "rms_db" in result
        assert "peak" in result
        assert "is_valid" in result
        assert "warnings" in result

    def test_validate_audio_short_duration(self):
        """짧은 오디오 검증 테스트"""
        from tools.prepare_voice_sample import validate_audio

        # 1초 미만의 짧은 오디오
        short_audio = np.random.randn(12000).astype(np.float32) * 0.5

        result = validate_audio(short_audio, sample_rate=24000)

        # 경고가 있어야 함
        assert len(result["warnings"]) > 0

    def test_generate_sample_script(self):
        """녹음 스크립트 생성 테스트"""
        from tools.prepare_voice_sample import generate_sample_script

        script = generate_sample_script()

        assert isinstance(script, str)
        assert len(script) > 0
        assert "안녕하세요" in script


class TestGenerateIdleLoops:
    """Idle 루프 생성 테스트"""

    def test_emotion_configs_exist(self):
        """감정 설정이 존재하는지 테스트"""
        from tools.generate_idle_loops import EMOTION_CONFIGS

        expected_emotions = ["neutral", "happy", "sad", "listening", "thinking"]

        for emotion in expected_emotions:
            assert emotion in EMOTION_CONFIGS, f"Missing emotion config: {emotion}"

    def test_emotion_configs_have_required_fields(self):
        """감정 설정에 필수 필드가 있는지 테스트"""
        from tools.generate_idle_loops import EMOTION_CONFIGS

        required_fields = ["filename", "blink_frequency", "head_movement"]

        for emotion, config in EMOTION_CONFIGS.items():
            for field in required_fields:
                assert field in config, f"Missing field '{field}' in {emotion}"

    def test_idle_loop_generator_initialization(self):
        """IdleLoopGenerator 초기화 테스트"""
        from tools.generate_idle_loops import IdleLoopGenerator

        generator = IdleLoopGenerator(
            output_size=(256, 256),
            fps=30,
            duration=2.0,
        )

        assert generator.output_size == (256, 256)
        assert generator.fps == 30
        assert generator.total_frames == 60

    def test_apply_animation(self):
        """애니메이션 적용 테스트"""
        from tools.generate_idle_loops import IdleLoopGenerator, EMOTION_CONFIGS

        generator = IdleLoopGenerator(output_size=(256, 256), fps=30, duration=2.0)

        # 테스트 이미지
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        config = EMOTION_CONFIGS["neutral"]

        animated = generator._apply_animation(test_image.copy(), t=0.5, config=config)

        assert isinstance(animated, np.ndarray)
        assert animated.shape == test_image.shape


class TestBenchmark:
    """벤치마크 도구 테스트"""

    def test_benchmark_runner_initialization(self):
        """BenchmarkRunner 초기화 테스트"""
        from tools.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(
            iterations=5,
            warmup_iterations=1,
            device="cpu",
        )

        assert runner.iterations == 5
        assert runner.warmup_iterations == 1
        assert runner.device == "cpu"

    def test_generate_test_audio(self):
        """테스트 오디오 생성 테스트"""
        from tools.benchmark import BenchmarkRunner

        runner = BenchmarkRunner()

        audio = runner._generate_test_audio(duration_sec=1.0, sample_rate=16000)

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) == 16000

    def test_calculate_stats(self):
        """통계 계산 테스트"""
        from tools.benchmark import BenchmarkRunner

        runner = BenchmarkRunner()

        times = [10.0, 12.0, 11.0, 13.0, 9.0, 10.0, 11.0, 12.0, 10.0, 11.0]
        stats = runner._calculate_stats(times, "Test")

        assert "name" in stats
        assert "count" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "p50" in stats
        assert "p90" in stats

        assert stats["count"] == 10
        assert stats["min"] == 9.0
        assert stats["max"] == 13.0

    def test_calculate_stats_empty(self):
        """빈 시간 리스트 통계 계산 테스트"""
        from tools.benchmark import BenchmarkRunner

        runner = BenchmarkRunner()

        stats = runner._calculate_stats([], "Test")

        assert "error" in stats
