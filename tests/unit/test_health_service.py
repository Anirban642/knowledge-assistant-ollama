from app.services.health_service import HealthService
from tests.fakes import FakeChunkRepository, FakeOllamaStatusChecker


def build_service(checker: FakeOllamaStatusChecker, repository: FakeChunkRepository) -> HealthService:
    return HealthService(
        ollama_status_checker=checker,
        chunk_repository=repository,
        required_models=["llama3", "nomic-embed-text"],
    )


def by_name(checks):
    return {check.name: check for check in checks}


def test_everything_healthy():
    checks = build_service(FakeOllamaStatusChecker(), FakeChunkRepository()).check_readiness()

    assert [check.name for check in checks] == ["ollama", "model:llama3", "model:nomic-embed-text", "vector_store"]
    assert all(check.healthy for check in checks)


def test_unreachable_ollama_is_reported_and_model_checks_are_skipped():
    checks = by_name(build_service(FakeOllamaStatusChecker(is_down=True), FakeChunkRepository()).check_readiness())

    assert checks["ollama"].healthy is False
    assert "model:llama3" not in checks
    assert checks["vector_store"].healthy is True


def test_missing_model_tells_the_user_how_to_install_it():
    checker = FakeOllamaStatusChecker(models={"llama3:latest"})

    checks = by_name(build_service(checker, FakeChunkRepository()).check_readiness())

    assert checks["model:nomic-embed-text"].healthy is False
    assert "ollama pull nomic-embed-text" in checks["model:nomic-embed-text"].detail


def test_a_similarly_named_model_does_not_count_as_installed():
    checker = FakeOllamaStatusChecker(models={"llama3.1:latest", "nomic-embed-text:latest"})

    checks = by_name(build_service(checker, FakeChunkRepository()).check_readiness())

    assert checks["model:llama3"].healthy is False


def test_vector_store_outage_is_reported():
    repository = FakeChunkRepository()
    repository.is_down = True

    checks = by_name(build_service(FakeOllamaStatusChecker(), repository).check_readiness())

    assert checks["vector_store"].healthy is False