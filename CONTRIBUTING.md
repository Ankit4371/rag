# Contributing to Advanced RAG

First off, thank you for considering contributing to this project!

## Development Setup

1. **Install `uv`**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Setup Environment**:
   ```bash
   make install
   cp .env.template .env
   ```

3. **Run Tests**:
   ```bash
   make test
   ```

## Workflow

1. Fork the repo and create your branch from `main`.
2. Ensure your code follows the project's style (we use `ruff` for linting).
3. Add tests for any new features or bug fixes.
4. Open a Pull Request with a clear description of your changes.

## Advanced RAG Implementation Guidelines

When adding new retrieval techniques:
- **Offload Compute**: Favor cloud APIs (Groq, Cohere, Qdrant) over local heavy models.
- **Trace Everything**: Ensure new steps are added to the `RAGPipeline` trace for UI visibility.
- **Maintain Safety**: Wrap new generation or retrieval logic with the existing guardrail system.
- **Evaluate**: Run the RAGAS suite in `evals/` to ensure no regression in faithfulness or relevancy.

## Code of Conduct

Please be respectful and professional in all interactions.
