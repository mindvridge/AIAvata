"""
LLM Module for conversation response generation.

Claude API / OpenAI GPT 기반 대화 응답 생성 모듈
특징:
- 스트리밍 응답 지원
- 대화 히스토리 관리
- 시스템 프롬프트 관리
"""

import asyncio
import logging
from typing import AsyncGenerator, List, Optional, Literal

from ..models.schemas import LLMResponse

logger = logging.getLogger(__name__)


class LLMModule:
    """
    LLM 대화 응답 생성 모듈

    Features:
    - Claude API / OpenAI GPT 지원
    - 스트리밍 응답
    - 대화 컨텍스트 관리
    - 감정 인식 연동
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        provider: Literal["anthropic", "openai"] = "anthropic",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        """
        Initialize LLM Module.

        Args:
            api_key: API key for the LLM provider
            model: Model name to use
            provider: LLM provider ('anthropic' or 'openai')
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        """
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature

        self._client = None
        self._initialized = False

    async def initialize(self) -> None:
        """클라이언트 초기화"""
        if self._initialized:
            return

        logger.info(f"Initializing LLM module with provider: {self.provider}")

        try:
            if self.provider == "anthropic":
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic(api_key=self.api_key)
            else:  # openai
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(api_key=self.api_key)

            self._initialized = True
            logger.info("LLM module initialized successfully")

        except ImportError as e:
            logger.error(f"Required package not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise

    def _ensure_initialized(self) -> None:
        """초기화 확인"""
        if not self._initialized:
            raise RuntimeError("LLM module not initialized. Call initialize() first.")

    async def generate(
        self,
        user_message: str,
        system_prompt: str = "",
        conversation_history: Optional[List[dict]] = None,
        user_emotion: Optional[str] = None,
    ) -> str:
        """
        단일 응답 생성 (비스트리밍)

        Args:
            user_message: 사용자 메시지
            system_prompt: 시스템 프롬프트
            conversation_history: 이전 대화 내역
            user_emotion: 감지된 사용자 감정

        Returns:
            생성된 응답 텍스트
        """
        self._ensure_initialized()

        # 감정 정보를 시스템 프롬프트에 추가
        enhanced_prompt = self._enhance_system_prompt(system_prompt, user_emotion)

        # 메시지 구성
        messages = self._build_messages(
            user_message, conversation_history or []
        )

        try:
            if self.provider == "anthropic":
                response = await self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=enhanced_prompt,
                    messages=messages,
                    temperature=self.temperature,
                )
                return response.content[0].text

            else:  # openai
                full_messages = [{"role": "system", "content": enhanced_prompt}]
                full_messages.extend(messages)

                response = await self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=full_messages,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content

        except Exception as e:
            error_msg = self._get_error_message(e)
            logger.error(f"LLM generation error: {type(e).__name__}: {e}")
            return error_msg

    async def generate_stream(
        self,
        user_message: str,
        system_prompt: str = "",
        conversation_history: Optional[List[dict]] = None,
        user_emotion: Optional[str] = None,
        chunk_size: int = 10,
    ) -> AsyncGenerator[LLMResponse, None]:
        """
        스트리밍 응답 생성

        Args:
            user_message: 사용자 메시지
            system_prompt: 시스템 프롬프트
            conversation_history: 이전 대화 내역
            user_emotion: 감지된 사용자 감정
            chunk_size: 청크당 최소 문자 수 (TTS 효율성을 위해)

        Yields:
            LLMResponse: 응답 청크
        """
        self._ensure_initialized()

        # 감정 정보를 시스템 프롬프트에 추가
        enhanced_prompt = self._enhance_system_prompt(system_prompt, user_emotion)

        # 메시지 구성
        messages = self._build_messages(
            user_message, conversation_history or []
        )

        try:
            if self.provider == "anthropic":
                async for response in self._stream_anthropic(
                    enhanced_prompt, messages, chunk_size
                ):
                    yield response
            else:
                async for response in self._stream_openai(
                    enhanced_prompt, messages, chunk_size
                ):
                    yield response

        except Exception as e:
            error_msg = self._get_error_message(e)
            logger.error(f"LLM streaming error: {type(e).__name__}: {e}")
            yield LLMResponse(
                text=error_msg,
                is_complete=True,
                token_count=0,
            )

    async def _stream_anthropic(
        self,
        system_prompt: str,
        messages: List[dict],
        chunk_size: int,
    ) -> AsyncGenerator[LLMResponse, None]:
        """Anthropic Claude API 스트리밍"""
        buffer = ""
        total_tokens = 0

        async with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=messages,
            temperature=self.temperature,
        ) as stream:
            async for text in stream.text_stream:
                buffer += text
                total_tokens += 1

                # 문장 단위 또는 최소 청크 크기로 yield
                if self._should_yield_chunk(buffer, chunk_size):
                    yield LLMResponse(
                        text=buffer,
                        is_complete=False,
                        token_count=total_tokens,
                    )
                    buffer = ""

        # 남은 버퍼 처리
        if buffer:
            yield LLMResponse(
                text=buffer,
                is_complete=True,
                token_count=total_tokens,
            )
        else:
            yield LLMResponse(
                text="",
                is_complete=True,
                token_count=total_tokens,
            )

    async def _stream_openai(
        self,
        system_prompt: str,
        messages: List[dict],
        chunk_size: int,
    ) -> AsyncGenerator[LLMResponse, None]:
        """OpenAI GPT API 스트리밍"""
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)

        buffer = ""
        total_tokens = 0

        stream = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=full_messages,
            temperature=self.temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                buffer += text
                total_tokens += 1

                if self._should_yield_chunk(buffer, chunk_size):
                    yield LLMResponse(
                        text=buffer,
                        is_complete=False,
                        token_count=total_tokens,
                    )
                    buffer = ""

        # 남은 버퍼 처리
        if buffer:
            yield LLMResponse(
                text=buffer,
                is_complete=True,
                token_count=total_tokens,
            )
        else:
            yield LLMResponse(
                text="",
                is_complete=True,
                token_count=total_tokens,
            )

    def _should_yield_chunk(self, buffer: str, min_size: int) -> bool:
        """
        청크를 yield할지 결정

        문장 종결 또는 최소 크기 도달 시 True
        """
        if len(buffer) < min_size:
            return False

        # 문장 종결 부호로 끝나는 경우
        sentence_enders = [".", "!", "?", "。", "！", "？", "\n"]
        for ender in sentence_enders:
            if buffer.rstrip().endswith(ender):
                return True

        # 쉼표 등 구두점으로 끝나고 충분히 긴 경우
        if len(buffer) >= min_size * 2:
            pause_markers = [",", ";", ":", "，", "；", "："]
            for marker in pause_markers:
                if buffer.rstrip().endswith(marker):
                    return True

        # 매우 긴 경우 강제 yield
        return len(buffer) >= min_size * 5

    def _enhance_system_prompt(
        self, base_prompt: str, user_emotion: Optional[str]
    ) -> str:
        """
        사용자 감정 정보로 시스템 프롬프트 강화

        Args:
            base_prompt: 기본 시스템 프롬프트
            user_emotion: 감지된 사용자 감정

        Returns:
            강화된 시스템 프롬프트
        """
        if not user_emotion or user_emotion == "neutral":
            return base_prompt

        emotion_guidance = {
            "happy": "사용자가 기쁜 상태입니다. 긍정적이고 활기찬 톤으로 대화해주세요.",
            "sad": "사용자가 슬픈 상태입니다. 공감하고 위로하는 톤으로 대화해주세요.",
            "angry": "사용자가 화난 상태입니다. 차분하고 이해하는 톤으로 대화하며, 문제 해결에 집중해주세요.",
            "surprised": "사용자가 놀란 상태입니다. 상황을 명확히 설명하고 안심시켜주세요.",
            "fearful": "사용자가 두려워하는 상태입니다. 안심시키고 도움이 되는 정보를 제공해주세요.",
        }

        guidance = emotion_guidance.get(user_emotion, "")
        if guidance:
            return f"{base_prompt}\n\n[감정 인식] {guidance}"
        return base_prompt

    def _build_messages(
        self, user_message: str, history: List[dict]
    ) -> List[dict]:
        """
        API 형식에 맞는 메시지 리스트 구성

        Args:
            user_message: 현재 사용자 메시지
            history: 대화 히스토리

        Returns:
            메시지 리스트
        """
        messages = []

        # 히스토리 추가 (최근 10개로 제한)
        for msg in history[-10:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": content})

        # 현재 메시지 추가
        messages.append({"role": "user", "content": user_message})

        return messages

    def _get_error_message(self, error: Exception) -> str:
        """에러 유형에 따른 사용자 친화적 메시지 반환"""
        error_type = type(error).__name__
        error_str = str(error).lower()

        # API 인증 오류
        if "auth" in error_str or "api_key" in error_str or "401" in error_str:
            return "죄송합니다. 서비스 인증에 문제가 있습니다. 관리자에게 문의해주세요."

        # 요청 한도 초과
        if "rate" in error_str or "limit" in error_str or "429" in error_str:
            return "죄송합니다. 요청이 너무 많습니다. 잠시 후 다시 시도해주세요."

        # 네트워크 오류
        if "connection" in error_str or "timeout" in error_str or "network" in error_str:
            return "죄송합니다. 네트워크 연결에 문제가 있습니다. 잠시 후 다시 시도해주세요."

        # 콘텐츠 필터링
        if "content" in error_str and ("filter" in error_str or "policy" in error_str):
            return "죄송합니다. 해당 내용은 처리할 수 없습니다."

        # 기본 오류 메시지
        return "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."

    async def cleanup(self) -> None:
        """리소스 정리"""
        if self._client:
            # Anthropic/OpenAI 클라이언트는 명시적 정리 불필요
            self._client = None
        self._initialized = False
        logger.info("LLM module cleaned up")
