"""Optional AI summarization via Ollama (local) or OpenAI (cloud)."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a legal document analyst. Summarize the following text from the "
    "Epstein case files in plain English. Be factual and concise. Include "
    "key names, dates, and actions mentioned. Do not speculate."
)


class Summarizer:
    """Generate short summaries of document text.

    Supports two providers:
    - ``"ollama"`` -- calls a local Ollama instance at ``localhost:11434``
    - ``"openai"`` -- calls the OpenAI chat completions API (requires the
      ``openai`` package and ``OPENAI_API_KEY`` environment variable)

    If the chosen provider is unavailable the summarizer falls back to
    returning the first *max_length* characters of the input text, so
    callers never need to handle failure.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.2",
    ) -> None:
        self.provider = provider.lower().strip()
        self.model = model

        # Eagerly validate the openai provider so we fail fast if the
        # package is missing.
        self._openai_client = None
        if self.provider == "openai":
            try:
                import openai  # noqa: F811

                self._openai_client = openai.OpenAI()
            except ImportError:
                logger.warning(
                    "openai package not installed -- summarizer will use fallback"
                )
            except Exception as exc:
                logger.warning("Failed to initialise OpenAI client: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(self, text: str, max_length: int = 200) -> str:
        """Return a short summary of *text*.

        Parameters
        ----------
        text:
            The full document text to summarize.
        max_length:
            Target maximum character length for the summary.

        Returns
        -------
        str
            A concise summary.  On provider failure the first *max_length*
            characters of *text* are returned as a best-effort fallback.
        """
        if not text or not text.strip():
            return ""

        # Truncate very long inputs to avoid excessive token usage.
        # 12,000 chars is roughly 3-4K tokens.
        truncated = text[:12_000]
        user_prompt = (
            f"Summarize the following in {max_length} characters or fewer:\n\n"
            f"{truncated}"
        )

        try:
            if self.provider == "ollama":
                return self._summarize_ollama(user_prompt, max_length)
            elif self.provider == "openai":
                return self._summarize_openai(user_prompt, max_length)
            else:
                logger.warning("Unknown provider %r, using fallback", self.provider)
                return self._fallback(text, max_length)
        except Exception as exc:
            logger.warning("Summarization failed (%s), using fallback: %s", self.provider, exc)
            return self._fallback(text, max_length)

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _summarize_ollama(self, user_prompt: str, max_length: int) -> str:
        """Call a local Ollama instance."""
        response = httpx.post(
            "http://localhost:11434/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "num_predict": max_length,
                    "temperature": 0.3,
                },
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "")
        return content.strip()[:max_length]

    def _summarize_openai(self, user_prompt: str, max_length: int) -> str:
        """Call the OpenAI chat completions API."""
        if self._openai_client is None:
            return self._fallback(user_prompt, max_length)

        response = self._openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_length,
            temperature=0.3,
        )
        content = response.choices[0].message.content or ""
        return content.strip()[:max_length]

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback(text: str, max_length: int) -> str:
        """Return the first *max_length* chars, trimmed at a word boundary."""
        cleaned = " ".join(text.split())
        if len(cleaned) <= max_length:
            return cleaned
        # Trim at the last space before the limit to avoid mid-word cuts.
        cut = cleaned[:max_length].rsplit(" ", 1)[0]
        return cut + "..."
