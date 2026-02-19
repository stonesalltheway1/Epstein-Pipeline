# Contributing to Epstein Pipeline

Thank you for your interest in contributing to this public interest research project. The Epstein Pipeline makes case files accessible and searchable for journalists, researchers, and the public.

## Quick Links

- [Open an Issue](https://github.com/stonesalltheway1/Epstein-Pipeline/issues/new/choose)
- [Project Architecture](docs/ARCHITECTURE.md)
- [Data Sources](docs/DATA_SOURCES.md)
- [Processor Reference](docs/PROCESSORS.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md)

## Ways to Contribute

### No Coding Required

| Contribution | How |
|---|---|
| **Report data quality issues** | Found a wrong person match, misclassified document, or bad OCR? [Open an issue](https://github.com/stonesalltheway1/Epstein-Pipeline/issues/new?template=bug-report.md). |
| **Suggest new data sources** | Know of a public dataset we're missing? [Use the data source template](https://github.com/stonesalltheway1/Epstein-Pipeline/issues/new?template=new-data-source.md). |
| **Person corrections** | Wrong name spelling, missing aliases, or incorrect person linking? [Use the person correction template](https://github.com/stonesalltheway1/Epstein-Pipeline/issues/new?template=person-correction.md). |
| **Review processed data** | Check `data/known-duplicates.json` and verify dedup decisions are correct. |
| **Improve documentation** | Fix typos, clarify instructions, add examples, translate docs. |

### Code Contributions

| Area | Description | Key Files |
|---|---|---|
| **New downloaders** | Support a new public data source | `src/epstein_pipeline/downloaders/` |
| **OCR improvements** | Better text extraction, new backends | `src/epstein_pipeline/processors/ocr.py` |
| **Entity extraction** | Better person matching, new entity types | `src/epstein_pipeline/processors/entities.py` |
| **Export formats** | New output formats for researchers | `src/epstein_pipeline/exporters/` |
| **Deduplication** | Better duplicate detection logic | `src/epstein_pipeline/processors/dedup.py` |
| **Knowledge graph** | Relationship extraction improvements | `src/epstein_pipeline/processors/knowledge_graph.py` |
| **Tests** | Increase coverage, add edge cases | `tests/` |

## Getting Started

### Prerequisites

- Python 3.10+ (3.12 recommended)
- Git
- ~2 GB disk space for dependencies

### Setup

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Epstein-Pipeline.git
cd Epstein-Pipeline

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Install in development mode with all dev tools
pip install -e ".[dev]"

# 4. Verify your setup
pytest -v
ruff check src/ tests/
```

### Optional: Install Processing Backends

Only install what you need for your contribution:

```bash
# OCR development (CPU)
pip install -e ".[ocr,ocr-surya,pymupdf]"

# NLP / entity extraction
pip install -e ".[nlp,nlp-gliner]"
python -m spacy download en_core_web_sm

# Embeddings / semantic search
pip install -e ".[embeddings]"

# Neon Postgres export
pip install -e ".[neon]"

# Everything (except GPU-only olmOCR)
pip install -e ".[all]"
```

### Docker Development

If you prefer Docker:

```bash
docker compose build
docker compose run pipeline --help
docker compose run pipeline ocr ./test-pdfs/ --output ./output/
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/my-improvement
```

Use descriptive branch names: `fix/ocr-timeout`, `feature/new-exporter`, `docs/api-reference`.

### 2. Make Changes

- Write code that follows the existing patterns in the codebase
- Add or update tests for your changes
- Update documentation if you changed behavior

### 3. Run Quality Checks

All three must pass before submitting a PR:

```bash
# Lint (catches bugs, enforces style)
ruff check src/ tests/

# Format (consistent code style)
ruff format src/ tests/

# Tests (verify nothing is broken)
pytest -v

# Optional: type checking
mypy src/epstein_pipeline/ --ignore-missing-imports
```

### 4. Commit with Clear Messages

```
Add Surya OCR backend with confidence scoring

- Implement SuryaOCRBackend in processors/ocr.py
- Add per-page confidence scores (0.0-1.0)
- Fall back to Docling when Surya confidence < 0.5
- Add tests for multi-page PDFs with mixed quality
```

### 5. Push and Open a Pull Request

```bash
git push origin feature/my-improvement
```

Then open a PR on GitHub. The CI pipeline will automatically:
- Run `ruff check` and `ruff format --check`
- Run `pytest` on Python 3.10, 3.11, 3.12, and 3.13
- Run `mypy` type checking
- Validate the Neon schema

## Code Standards

### Style

- **Formatter/Linter:** [Ruff](https://github.com/astral-sh/ruff) (configured in `pyproject.toml`)
- **Line length:** 100 characters
- **Imports:** Sorted by Ruff (`isort` rules)
- **Naming:** `snake_case` for functions/variables, `PascalCase` for classes
  - Exception: Pydantic model fields use `camelCase` to match the site's TypeScript interfaces (hence `N815` is ignored in Ruff config)

### Type Hints

- Encouraged for all public functions
- Required for Pydantic models
- Use `from __future__ import annotations` for modern syntax

### Docstrings

- Required for public functions and classes
- Use Google-style docstrings:

```python
def extract_entities(text: str, model: str = "en_core_web_sm") -> list[Entity]:
    """Extract named entities from document text.

    Args:
        text: Raw document text from OCR.
        model: spaCy model name to use for NER.

    Returns:
        List of extracted entities with confidence scores.
    """
```

### Testing

- Tests go in `tests/` with filenames matching `test_*.py`
- Use pytest fixtures from `tests/conftest.py` for shared setup
- Aim for testing edge cases: empty text, Unicode, huge documents, malformed PDFs
- Mark slow tests: `@pytest.mark.slow`

```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_entities.py -v

# Run with coverage report
pytest --cov=epstein_pipeline --cov-report=term-missing
```

## Submitting Processed Data

If you've run the pipeline on new documents:

1. Export as JSON: `epstein-pipeline export json ./processed/ --output ./contribution/`
2. Validate: `epstein-pipeline validate ./contribution/`
3. Open a PR with the JSON files in a `contributions/` directory
4. CI will automatically validate the data schema

## Project Structure

```
Epstein-Pipeline/
├── src/epstein_pipeline/
│   ├── cli.py              # Click CLI entry point
│   ├── config.py           # Pydantic settings (env vars, paths)
│   ├── state.py            # Pipeline state tracking
│   ├── downloaders/        # Data source fetchers (DOJ, Kaggle, HF, Archive.org)
│   ├── processors/         # Core processing (OCR, NER, dedup, embeddings, etc.)
│   ├── exporters/          # Output formats (JSON, CSV, SQLite, Neon Postgres)
│   ├── importers/          # External data importers (Sea_Doughnut)
│   ├── models/             # Pydantic data models
│   ├── validators/         # Schema and integrity checks
│   └── utils/              # Hashing, parallelism, progress bars
├── tests/                  # pytest test suite (19 test files)
├── data/                   # Person registry, known duplicates
├── docs/                   # Extended documentation
├── scripts/                # Utility scripts
├── .github/
│   ├── workflows/          # CI, publish, data validation
│   └── ISSUE_TEMPLATE/     # Bug report, data source, person correction
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Docker Compose for local dev
└── pyproject.toml          # Project config, deps, tool settings
```

## Review Process

1. A maintainer will review your PR within a few days
2. CI must pass (lint, format, tests, typecheck, schema validation)
3. For data changes, we verify accuracy against source documents
4. For code changes, we check for edge cases and test coverage
5. Once approved, a maintainer will merge and the change ships in the next release

## Getting Help

- **Questions:** Open a [Discussion](https://github.com/stonesalltheway1/Epstein-Pipeline/discussions) or ask in an issue
- **Bugs:** Use the [bug report template](https://github.com/stonesalltheway1/Epstein-Pipeline/issues/new?template=bug-report.md)
- **Live site issues:** Report at [epsteinexposed.com/contribute](https://epsteinexposed.com/contribute)

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
