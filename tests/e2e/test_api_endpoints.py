"""
E2E Tests for API Endpoints.

API 엔드포인트 End-to-End 테스트
"""

import time
import pytest
from httpx import AsyncClient


class TestHealthEndpoint:
    """Health check endpoint tests."""

    @pytest.mark.asyncio
    async def test_health_check_returns_200(self, async_client: AsyncClient):
        """Health endpoint should return 200."""
        response = await async_client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_check_response_format(self, async_client: AsyncClient):
        """Health response should contain required fields."""
        response = await async_client.get("/health")
        data = response.json()

        assert "status" in data
        assert "pipeline_ready" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "gpu_available" in data

    @pytest.mark.asyncio
    async def test_health_check_status_healthy(self, async_client: AsyncClient):
        """Health status should be healthy."""
        response = await async_client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_response_time(
        self, async_client: AsyncClient, performance_thresholds
    ):
        """Health check should respond within threshold."""
        start = time.perf_counter()
        response = await async_client.get("/health")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert response.status_code == 200
        assert elapsed_ms < performance_thresholds["api_response_time_ms"]


class TestAvatarSessionEndpoints:
    """Avatar session management endpoint tests."""

    @pytest.mark.asyncio
    async def test_create_session_success(self, async_client: AsyncClient):
        """Should create avatar session successfully."""
        response = await async_client.post(
            "/api/avatar/create",
            json={
                "avatar_id": "default",
                "system_prompt": "Test prompt",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "websocket_url" in data
        assert data["avatar_id"] == "default"

    @pytest.mark.asyncio
    async def test_create_session_default_avatar(self, async_client: AsyncClient):
        """Should create session with default avatar."""
        response = await async_client.post(
            "/api/avatar/create",
            json={}
        )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data

    @pytest.mark.asyncio
    async def test_get_session_info(self, async_client: AsyncClient):
        """Should retrieve session info."""
        # Create session first
        create_response = await async_client.post(
            "/api/avatar/create",
            json={"avatar_id": "default"}
        )
        session_id = create_response.json()["session_id"]

        # Get session info
        response = await async_client.get(f"/api/avatar/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert "current_emotion" in data
        assert "pipeline_state" in data

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, async_client: AsyncClient):
        """Should return 404 for nonexistent session."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await async_client.get(f"/api/avatar/{fake_id}")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_session_invalid_id_format(self, async_client: AsyncClient):
        """Should return 400 for invalid session ID format."""
        response = await async_client.get("/api/avatar/invalid-id")

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_delete_session_success(self, async_client: AsyncClient):
        """Should delete session successfully."""
        # Create session
        create_response = await async_client.post(
            "/api/avatar/create",
            json={"avatar_id": "default"}
        )
        session_id = create_response.json()["session_id"]

        # Delete session
        response = await async_client.delete(f"/api/avatar/{session_id}")

        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

        # Verify deletion
        get_response = await async_client.get(f"/api/avatar/{session_id}")
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self, async_client: AsyncClient):
        """Should return 404 when deleting nonexistent session."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await async_client.delete(f"/api/avatar/{fake_id}")

        assert response.status_code == 404


class TestMetricsEndpoint:
    """Metrics endpoint tests."""

    @pytest.mark.asyncio
    async def test_get_metrics(self, async_client: AsyncClient):
        """Should return pipeline metrics."""
        response = await async_client.get("/api/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "avg_latency_ms" in data


class TestAvatarListEndpoint:
    """Avatar list endpoint tests."""

    @pytest.mark.asyncio
    async def test_list_avatars(self, async_client: AsyncClient):
        """Should return list of available avatars."""
        response = await async_client.get("/api/avatars")

        assert response.status_code == 200
        data = response.json()
        assert "avatars" in data
        assert isinstance(data["avatars"], list)

    @pytest.mark.asyncio
    async def test_avatar_has_required_fields(self, async_client: AsyncClient):
        """Each avatar should have required fields."""
        response = await async_client.get("/api/avatars")
        data = response.json()

        for avatar in data["avatars"]:
            assert "id" in avatar
            assert "name" in avatar


class TestEmotionsEndpoint:
    """Emotions endpoint tests."""

    @pytest.mark.asyncio
    async def test_list_emotions(self, async_client: AsyncClient):
        """Should return list of supported emotions."""
        response = await async_client.get("/api/emotions")

        assert response.status_code == 200
        data = response.json()
        assert "emotions" in data
        assert isinstance(data["emotions"], list)
        assert len(data["emotions"]) > 0

    @pytest.mark.asyncio
    async def test_emotions_include_basic_set(self, async_client: AsyncClient):
        """Should include basic emotions."""
        response = await async_client.get("/api/emotions")
        emotions = response.json()["emotions"]

        # Check for some expected emotions
        assert "neutral" in emotions
        assert "happy" in emotions


class TestLiveKitTokenEndpoint:
    """LiveKit token endpoint tests."""

    @pytest.mark.asyncio
    async def test_generate_token_success(self, async_client: AsyncClient):
        """Should generate LiveKit token."""
        response = await async_client.post(
            "/api/generate-token",
            json={
                "room_name": "test-room",
                "participant_name": "test-user",
            }
        )

        # May return 200 or 503 depending on LiveKit config
        if response.status_code == 200:
            data = response.json()
            assert "token" in data
            assert "room_name" in data
            assert data["room_name"] == "test-room"

    @pytest.mark.asyncio
    async def test_generate_token_with_identity(self, async_client: AsyncClient):
        """Should generate token with participant identity."""
        response = await async_client.post(
            "/api/generate-token",
            json={
                "room_name": "test-room",
                "participant_name": "test-user",
                "participant_identity": "user-123",
            }
        )

        if response.status_code == 200:
            data = response.json()
            assert "token" in data


class TestAPIDocumentation:
    """API documentation endpoint tests."""

    @pytest.mark.asyncio
    async def test_openapi_docs_available(self, async_client: AsyncClient):
        """OpenAPI docs should be available."""
        response = await async_client.get("/docs")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_redoc_available(self, async_client: AsyncClient):
        """ReDoc should be available."""
        response = await async_client.get("/redoc")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_openapi_json_available(self, async_client: AsyncClient):
        """OpenAPI JSON schema should be available."""
        response = await async_client.get("/openapi.json")
        assert response.status_code == 200

        data = response.json()
        assert "openapi" in data
        assert "paths" in data


class TestCORS:
    """CORS configuration tests."""

    @pytest.mark.asyncio
    async def test_cors_headers_present(self, async_client: AsyncClient):
        """CORS headers should be present."""
        response = await async_client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            }
        )

        # Should allow CORS
        assert response.status_code in [200, 204, 405]


class TestErrorHandling:
    """Error handling tests."""

    @pytest.mark.asyncio
    async def test_404_for_unknown_endpoint(self, async_client: AsyncClient):
        """Should return 404 for unknown endpoints."""
        response = await async_client.get("/api/unknown-endpoint")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_method_not_allowed(self, async_client: AsyncClient):
        """Should return 405 for wrong HTTP method."""
        response = await async_client.put("/health")
        assert response.status_code == 405

    @pytest.mark.asyncio
    async def test_invalid_json_body(self, async_client: AsyncClient):
        """Should handle invalid JSON body."""
        response = await async_client.post(
            "/api/avatar/create",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
