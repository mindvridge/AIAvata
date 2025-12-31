#!/usr/bin/env python3
"""
Performance Benchmark Tool.

각 파이프라인 컴포넌트의 성능을 측정합니다.

사용법:
    python tools/benchmark.py
    python tools/benchmark.py --component stt
    python tools/benchmark.py --iterations 100
"""

import argparse
import asyncio
import logging
import statistics
import sys
import time
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """벤치마크 실행기"""

    def __init__(
        self,
        iterations: int = 10,
        warmup_iterations: int = 2,
        device: str = "cuda",
    ):
        """
        Initialize Benchmark Runner.

        Args:
            iterations: 측정 반복 횟수
            warmup_iterations: 워밍업 반복 횟수
            device: 사용할 디바이스
        """
        self.iterations = iterations
        self.warmup_iterations = warmup_iterations
        self.device = device

        self.results: Dict[str, List[float]] = {}

    def _generate_test_audio(
        self,
        duration_sec: float = 3.0,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """테스트용 오디오 생성"""
        samples = int(duration_sec * sample_rate)
        # 간단한 사인파 + 노이즈
        t = np.linspace(0, duration_sec, samples)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz
        audio += 0.1 * np.random.randn(samples)  # 노이즈
        return audio.astype(np.float32)

    async def benchmark_stt(self) -> Dict[str, float]:
        """STT 벤치마크"""
        logger.info("Benchmarking STT module...")

        try:
            from src.pipeline.stt_module import STTModule

            stt = STTModule(device=self.device)
            await stt.initialize()

            # 테스트 오디오 생성
            test_audio = self._generate_test_audio(3.0)

            times = []

            # 워밍업
            for _ in range(self.warmup_iterations):
                await stt.transcribe(test_audio)

            # 측정
            for i in range(self.iterations):
                start = time.perf_counter()
                result = await stt.transcribe(test_audio)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
                logger.debug(f"  Iteration {i + 1}: {elapsed:.2f}ms")

            await stt.cleanup()

            self.results["stt"] = times
            return self._calculate_stats(times, "STT")

        except Exception as e:
            logger.error(f"STT benchmark failed: {e}")
            return {"error": str(e)}

    async def benchmark_tts(self) -> Dict[str, float]:
        """TTS 벤치마크"""
        logger.info("Benchmarking TTS module...")

        try:
            from src.pipeline.tts_module import TTSModule

            tts = TTSModule(sample_rate=24000, device=self.device)
            await tts.initialize()

            # 테스트 텍스트
            test_texts = [
                "안녕하세요. 반갑습니다.",
                "오늘 날씨가 정말 좋네요.",
                "무엇을 도와드릴까요?",
            ]

            times = []

            # 워밍업
            for text in test_texts[:self.warmup_iterations]:
                await tts.synthesize(text)

            # 측정
            for i in range(self.iterations):
                text = test_texts[i % len(test_texts)]
                start = time.perf_counter()
                audio = await tts.synthesize(text)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

                # RTF (Real-Time Factor) 계산
                audio_duration = len(audio) / 24000 * 1000
                rtf = elapsed / audio_duration if audio_duration > 0 else 0
                logger.debug(f"  Iteration {i + 1}: {elapsed:.2f}ms (RTF: {rtf:.3f})")

            await tts.cleanup()

            self.results["tts"] = times
            return self._calculate_stats(times, "TTS")

        except Exception as e:
            logger.error(f"TTS benchmark failed: {e}")
            return {"error": str(e)}

    async def benchmark_llm(self) -> Dict[str, float]:
        """LLM 벤치마크"""
        logger.info("Benchmarking LLM module...")

        try:
            import os
            from src.pipeline.llm_module import LLMModule

            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not set, skipping LLM benchmark")
                return {"skipped": True}

            llm = LLMModule(
                api_key=api_key,
                model="claude-sonnet-4-20250514",
                provider="anthropic",
            )
            await llm.initialize()

            # 테스트 프롬프트
            test_prompts = [
                "안녕하세요?",
                "오늘 날씨 어때요?",
                "간단한 인사 한마디 해주세요.",
            ]

            times = []
            first_token_times = []

            # 워밍업
            for prompt in test_prompts[:self.warmup_iterations]:
                await llm.generate(prompt)

            # 측정 (스트리밍)
            for i in range(min(self.iterations, 5)):  # API 호출 제한
                prompt = test_prompts[i % len(test_prompts)]
                start = time.perf_counter()
                first_token_received = False
                first_token_time = 0

                async for response in llm.generate_stream(prompt):
                    if not first_token_received:
                        first_token_time = (time.perf_counter() - start) * 1000
                        first_token_received = True

                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
                first_token_times.append(first_token_time)
                logger.debug(
                    f"  Iteration {i + 1}: {elapsed:.2f}ms "
                    f"(TTFT: {first_token_time:.2f}ms)"
                )

            await llm.cleanup()

            self.results["llm"] = times
            self.results["llm_ttft"] = first_token_times

            stats = self._calculate_stats(times, "LLM Total")
            ttft_stats = self._calculate_stats(first_token_times, "LLM TTFT")
            stats["ttft_mean"] = ttft_stats["mean"]
            stats["ttft_p50"] = ttft_stats["p50"]

            return stats

        except Exception as e:
            logger.error(f"LLM benchmark failed: {e}")
            return {"error": str(e)}

    async def benchmark_avatar_renderer(self) -> Dict[str, float]:
        """Avatar Renderer 벤치마크"""
        logger.info("Benchmarking Avatar Renderer...")

        try:
            from src.pipeline.avatar_renderer import AvatarRenderer

            renderer = AvatarRenderer(
                output_width=784,
                output_height=1176,
                target_fps=30,
                device=self.device,
            )
            await renderer.initialize()

            times = []

            # 워밍업
            for _ in range(self.warmup_iterations):
                renderer.get_idle_frame()

            # 프레임 렌더링 측정
            for i in range(self.iterations):
                start = time.perf_counter()
                frame = renderer.get_idle_frame()
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
                logger.debug(f"  Iteration {i + 1}: {elapsed:.2f}ms")

            await renderer.cleanup()

            self.results["renderer"] = times
            stats = self._calculate_stats(times, "Avatar Renderer")

            # FPS 계산
            mean_time = stats["mean"]
            if mean_time > 0:
                stats["estimated_fps"] = 1000 / mean_time

            return stats

        except Exception as e:
            logger.error(f"Avatar Renderer benchmark failed: {e}")
            return {"error": str(e)}

    async def benchmark_vad(self) -> Dict[str, float]:
        """VAD 벤치마크"""
        logger.info("Benchmarking VAD module...")

        try:
            from src.utils.vad import VoiceActivityDetector

            vad = VoiceActivityDetector()
            vad.initialize()

            # 테스트 오디오
            test_audio = self._generate_test_audio(0.5)

            times = []

            # 워밍업
            for _ in range(self.warmup_iterations):
                vad.is_speech(test_audio)

            # 측정
            for i in range(self.iterations):
                start = time.perf_counter()
                is_speech, prob = vad.is_speech(test_audio, return_probability=True)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
                logger.debug(f"  Iteration {i + 1}: {elapsed:.2f}ms (prob: {prob:.3f})")

            self.results["vad"] = times
            return self._calculate_stats(times, "VAD")

        except Exception as e:
            logger.error(f"VAD benchmark failed: {e}")
            return {"error": str(e)}

    async def benchmark_musetalk(self) -> Dict[str, float]:
        """MuseTalk 립싱크 벤치마크"""
        logger.info("Benchmarking MuseTalk model...")

        try:
            from src.models.integrations import MuseTalkModel

            model = MuseTalkModel(device=self.device, fp16=(self.device == "cuda"))
            await model.initialize()

            # 테스트 데이터 (avata_ani.mp4 크기: 784x1176)
            test_frame = np.random.randint(0, 255, (1176, 784, 3), dtype=np.uint8)
            test_audio = self._generate_test_audio(0.033)  # 1프레임 분량

            times = []

            # 워밍업
            for _ in range(self.warmup_iterations):
                await model.process_frame(test_frame, test_audio, 24000)

            # 측정
            for i in range(self.iterations):
                start = time.perf_counter()
                result = await model.process_frame(test_frame, test_audio, 24000)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
                logger.debug(f"  Iteration {i + 1}: {elapsed:.2f}ms")

            await model.cleanup()

            self.results["musetalk"] = times
            stats = self._calculate_stats(times, "MuseTalk")

            # FPS 계산
            if stats["mean"] > 0:
                stats["estimated_fps"] = 1000 / stats["mean"]

            return stats

        except Exception as e:
            logger.error(f"MuseTalk benchmark failed: {e}")
            return {"error": str(e)}

    async def benchmark_live_portrait(self) -> Dict[str, float]:
        """LivePortrait 벤치마크"""
        logger.info("Benchmarking LivePortrait model...")

        try:
            from src.models.integrations import LivePortraitModel

            model = LivePortraitModel(
                device=self.device,
                output_size=(784, 1176),
                fp16=(self.device == "cuda"),
            )
            await model.initialize()

            # 소스 이미지 설정 (avata_ani.mp4 크기)
            test_image = np.random.randint(0, 255, (1176, 784, 3), dtype=np.uint8)
            await model.extract_source_features(test_image)

            times = []

            # 워밍업
            for _ in range(self.warmup_iterations):
                await model.generate_frame({"head_yaw": 0.1})

            # 측정
            for i in range(self.iterations):
                start = time.perf_counter()
                frame = await model.generate_frame(
                    motion_params={"head_yaw": np.sin(i * 0.1) * 0.1}
                )
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
                logger.debug(f"  Iteration {i + 1}: {elapsed:.2f}ms")

            await model.cleanup()

            self.results["live_portrait"] = times
            stats = self._calculate_stats(times, "LivePortrait")

            # FPS 계산
            if stats["mean"] > 0:
                stats["estimated_fps"] = 1000 / stats["mean"]

            return stats

        except Exception as e:
            logger.error(f"LivePortrait benchmark failed: {e}")
            return {"error": str(e)}

    async def benchmark_full_pipeline(self) -> Dict[str, float]:
        """전체 파이프라인 벤치마크"""
        logger.info("Benchmarking full pipeline...")

        try:
            import os
            from src.config import get_settings
            from src.pipeline.orchestrator import PipelineOrchestrator

            settings = get_settings()
            if not settings.llm_api_key:
                logger.warning("API key not set, skipping full pipeline benchmark")
                return {"skipped": True}

            pipeline = PipelineOrchestrator(settings)
            await pipeline.initialize()

            # 테스트 오디오
            test_audio = self._generate_test_audio(2.0)
            test_audio_bytes = (test_audio * 32767).astype(np.int16).tobytes()

            times = []
            first_frame_times = []

            # 워밍업
            for _ in range(min(self.warmup_iterations, 1)):
                async for _ in pipeline.process_audio_input(test_audio_bytes):
                    pass

            # 측정
            for i in range(min(self.iterations, 3)):  # API 호출 제한
                start = time.perf_counter()
                first_frame_received = False
                first_frame_time = 0
                frame_count = 0

                async for frame in pipeline.process_audio_input(test_audio_bytes):
                    if not first_frame_received:
                        first_frame_time = (time.perf_counter() - start) * 1000
                        first_frame_received = True
                    frame_count += 1

                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
                first_frame_times.append(first_frame_time)
                logger.debug(
                    f"  Iteration {i + 1}: {elapsed:.2f}ms "
                    f"(TTFF: {first_frame_time:.2f}ms, frames: {frame_count})"
                )

            await pipeline.cleanup()

            self.results["pipeline"] = times
            self.results["pipeline_ttff"] = first_frame_times

            stats = self._calculate_stats(times, "Full Pipeline")
            ttff_stats = self._calculate_stats(first_frame_times, "Pipeline TTFF")
            stats["ttff_mean"] = ttff_stats["mean"]
            stats["ttff_p50"] = ttff_stats["p50"]

            return stats

        except Exception as e:
            logger.error(f"Full pipeline benchmark failed: {e}")
            return {"error": str(e)}

    def _calculate_stats(
        self,
        times: List[float],
        name: str,
    ) -> Dict[str, float]:
        """통계 계산"""
        if not times:
            return {"error": "No measurements"}

        sorted_times = sorted(times)
        n = len(sorted_times)

        return {
            "name": name,
            "count": n,
            "mean": statistics.mean(times),
            "std": statistics.stdev(times) if n > 1 else 0,
            "min": min(times),
            "max": max(times),
            "p50": sorted_times[int(n * 0.5)],
            "p90": sorted_times[int(n * 0.9)] if n >= 10 else sorted_times[-1],
            "p99": sorted_times[int(n * 0.99)] if n >= 100 else sorted_times[-1],
        }


def print_results(results: Dict[str, Dict]) -> None:
    """결과 출력"""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    for component, stats in results.items():
        if "error" in stats:
            print(f"\n{component.upper()}: ERROR - {stats['error']}")
            continue

        if "skipped" in stats:
            print(f"\n{component.upper()}: SKIPPED")
            continue

        name = stats.get("name", component.upper())
        print(f"\n{name}:")
        print(f"  Iterations: {stats['count']}")
        print(f"  Mean:       {stats['mean']:.2f} ms")
        print(f"  Std Dev:    {stats['std']:.2f} ms")
        print(f"  Min:        {stats['min']:.2f} ms")
        print(f"  Max:        {stats['max']:.2f} ms")
        print(f"  P50:        {stats['p50']:.2f} ms")
        print(f"  P90:        {stats['p90']:.2f} ms")

        if "estimated_fps" in stats:
            print(f"  Est. FPS:   {stats['estimated_fps']:.1f}")

        if "ttft_mean" in stats:
            print(f"  TTFT Mean:  {stats['ttft_mean']:.2f} ms")

        if "ttff_mean" in stats:
            print(f"  TTFF Mean:  {stats['ttff_mean']:.2f} ms")

    print("\n" + "=" * 70)


async def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Benchmark pipeline components",
    )

    parser.add_argument(
        "--component",
        "-c",
        nargs="+",
        choices=["stt", "tts", "llm", "renderer", "vad", "musetalk", "live_portrait", "pipeline", "all"],
        default=["all"],
        help="Components to benchmark",
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=10,
        help="Number of iterations (default: 10)",
    )
    parser.add_argument(
        "--device",
        "-d",
        choices=["cuda", "cpu", "mps"],
        default="cuda",
        help="Device to use (default: cuda)",
    )

    args = parser.parse_args()

    # GPU 확인
    try:
        import torch
        if args.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            args.device = "cpu"
    except ImportError:
        args.device = "cpu"

    # 벤치마크 실행
    runner = BenchmarkRunner(
        iterations=args.iterations,
        device=args.device,
    )

    components = args.component
    if "all" in components:
        components = ["stt", "tts", "vad", "renderer", "musetalk", "live_portrait", "llm", "pipeline"]

    results = {}

    for component in components:
        if component == "stt":
            results["stt"] = await runner.benchmark_stt()
        elif component == "tts":
            results["tts"] = await runner.benchmark_tts()
        elif component == "llm":
            results["llm"] = await runner.benchmark_llm()
        elif component == "renderer":
            results["renderer"] = await runner.benchmark_avatar_renderer()
        elif component == "vad":
            results["vad"] = await runner.benchmark_vad()
        elif component == "musetalk":
            results["musetalk"] = await runner.benchmark_musetalk()
        elif component == "live_portrait":
            results["live_portrait"] = await runner.benchmark_live_portrait()
        elif component == "pipeline":
            results["pipeline"] = await runner.benchmark_full_pipeline()

    # 결과 출력
    print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
