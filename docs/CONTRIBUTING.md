# Contributing to Epstein Pipeline

Thank you for your interest in contributing! This project helps make the Epstein case files accessible and searchable for public interest research.

## Ways to Contribute

### No Coding Required

- **Report data quality issues** — Found a wrong person match or duplicate? Open an issue.
- **Suggest new data sources** — Know of a public dataset we're missing? Let us know.
- **Review processed data** — Check `known-duplicates.json` and verify dedup decisions.
- **Improve documentation** — Fix typos, clarify instructions, add examples.

### Coding Contributions

- **Add a new downloader** — Support a new data source in `src/epstein_pipeline/downloaders/`.
- **Improve entity extraction** — Better person matching, new entity types.
- **Add export formats** — New output formats for researchers.
- **Fix bugs** — Check the issue tracker for open bugs.

## Getting Started

### Prerequisites

- Python 3.10+
- Git

### Setup

```bash
# Clone the repo
git clone https://github.com/stonesalltheway1/Epstein-Pipeline.git
cd Epstein-Pipeline

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev]"

# Run tests to verify setup
pytest -v
```

### For OCR Development

```bash
# Install OCR dependencies
pip install -e ".[ocr]"
```

### For NLP Development

```bash
# Install NLP dependencies
pip install -e ".[nlp]"
python -m spacy download en_core_web_sm
```

## Development Workflow

1. **Fork** the repo and create a branch: `git checkout -b my-feature`
2. **Make changes** and add tests
3. **Run checks:**
   ```bash
   ruff check src/ tests/      # Lint
   ruff format src/ tests/     # Format
   pytest -v                    # Test
   ```
4. **Commit** with a clear message
5. **Push** and open a Pull Request

## Code Style

- We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Line length: 100 characters
- Type hints are encouraged but not required
- Docstrings for public functions

## Submitting Processed Data

If you've run the pipeline on new documents and want to contribute the results:

1. Export as JSON: `epstein-pipeline export json ./processed/ --output ./contribution/`
2. Validate: `epstein-pipeline validate ./contribution/`
3. Open a PR with the JSON files in a `contributions/` directory
4. Our CI will automatically validate the data

## Code of Conduct

This is a public interest research project. We expect all contributors to:
- Be respectful and constructive
- Focus on facts and verifiable information
- Not engage in harassment or doxxing
- Respect the privacy of uninvolved individuals
