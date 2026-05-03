.PHONY: install test lint format run

install:
	uv sync --extra dev

test:
	pytest tests/ -v

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

run:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
