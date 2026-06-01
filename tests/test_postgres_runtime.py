from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_postgres_compose_uses_pgvector_and_ignored_local_storage():
    compose = (ROOT / "docker-compose.postgres.yml").read_text(encoding="utf-8")
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "pgvector/pgvector:pg16" in compose
    assert "container_name: chunking-docs-postgres" in compose
    assert 'POSTGRES_DB: "${POSTGRES_DB:-chunking_docs}"' in compose
    assert 'POSTGRES_USER: "${POSTGRES_USER:-chunking_docs}"' in compose
    assert 'POSTGRES_PASSWORD: "${POSTGRES_PASSWORD:-chunking_docs}"' in compose
    assert '"${POSTGRES_PORT:-5432}:5432"' in compose
    assert '"${POSTGRES_DATA_PATH:-./.local/postgres}:/var/lib/postgresql/data"' in compose
    assert "pg_isready -U $${POSTGRES_USER} -d $${POSTGRES_DB}" in compose
    assert ".local/" in gitignore


def test_postgres_readme_documents_local_pgvector_bootstrap():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "docker compose -f docker-compose.postgres.yml up -d" in readme
    assert "postgresql://chunking_docs:chunking_docs@localhost:5432/chunking_docs" in readme
    assert "`docker-compose.postgres.yml` starts a local PostgreSQL 16 service with pgvector" in readme
