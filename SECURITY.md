# Security Policy

## Scope

The Epstein Pipeline processes legal documents from public government releases (DOJ EFTA). While the source material is public record, the pipeline handles sensitive data including:

- Names of individuals (some of whom are victims or witnesses)
- Financial records and account numbers
- Legal proceedings and sealed document references
- Redacted content that may be partially recoverable

## Reporting Vulnerabilities

If you discover a security vulnerability in the pipeline code, **do not open a public issue**. Instead:

1. Email **security@epsteinexposed.com** with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
2. We will acknowledge receipt within 48 hours
3. We will provide a fix timeline within 7 days

## Data Handling Guidelines

### For Contributors

- **Never commit real victim names or PII** in test fixtures. Use synthetic data.
- **Never commit credentials** (API keys, database URLs, tokens). Use environment variables.
- **Never commit raw PDFs** to the repository. The `data/` directory is for metadata only.
- **Redaction analysis results** (`recovered_text`) must be handled with care — recovered text from under redactions may contain sensitive information that was intentionally hidden.

### For Pipeline Operators

- Store `EPSTEIN_NEON_DATABASE_URL` and other secrets in environment variables, never in config files
- Use the `.env` file (gitignored) for local development credentials
- When running OCR on new documents, review output for unintentional PII exposure before sharing
- The `--describe` flag on image extraction sends images to AI vision models — be aware of what you're uploading

## Dependencies

We monitor dependencies for known vulnerabilities via GitHub Dependabot. The pipeline uses:

- **httpx** for HTTP requests (preferred over requests for async support)
- **psycopg** for Postgres connections (binary wheels, no libpq compilation needed)
- **spaCy** for NLP (models downloaded separately, not bundled)

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x (current) | Yes |
| < 1.0 | No |

## Responsible Disclosure

We follow responsible disclosure practices. If a vulnerability affects users of the live site (epsteinexposed.com), we will:

1. Patch the vulnerability
2. Deploy the fix
3. Notify affected users if applicable
4. Publish a security advisory after the fix is deployed
