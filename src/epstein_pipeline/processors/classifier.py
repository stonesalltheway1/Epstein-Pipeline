"""Zero-shot document classification using transformer models.

Two backends:
  1. GLiClass-ModernBERT (default) — 50x faster, 8K context, state-of-the-art
  2. BART-MNLI (legacy fallback) — slower but well-tested

Classifies documents into categories matching the site's DocumentCategory type:
deposition, email, legal-filing, indictment, plea-agreement, financial,
communications, investigation, testimony, other.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from epstein_pipeline.config import Settings
from epstein_pipeline.models.document import Document, DocumentCategory

logger = logging.getLogger(__name__)

# Categories for zero-shot classification — maps to DocumentCategory type
CLASSIFICATION_LABELS: list[str] = [
    "legal document or court filing",
    "financial record or bank statement",
    "travel document or flight log",
    "email or personal communication",
    "law enforcement investigation report",
    "news article or media report",
    "government document or official record",
    "personal notes or diary entry",
    "medical record or health document",
    "property or real estate document",
    "corporate filing or business record",
    "intelligence report or surveillance record",
]

# Map classification labels to DocumentCategory values
_LABEL_TO_CATEGORY: dict[str, DocumentCategory] = {
    "legal document or court filing": "legal",
    "financial record or bank statement": "financial",
    "travel document or flight log": "travel",
    "email or personal communication": "communications",
    "law enforcement investigation report": "investigation",
    "news article or media report": "media",
    "government document or official record": "government",
    "personal notes or diary entry": "personal",
    "medical record or health document": "medical",
    "property or real estate document": "property",
    "corporate filing or business record": "corporate",
    "intelligence report or surveillance record": "intelligence",
}

# GLiClass models are identified by these prefixes
_GLICLASS_PREFIXES = ("knowledgator/gliclass",)


@dataclass
class ClassificationResult:
    """Result of classifying a single document."""

    document_id: str
    predicted_category: DocumentCategory
    confidence: float
    all_scores: dict[str, float]


class DocumentClassifier:
    """Classify documents into categories using zero-shot classification.

    Supports two backends:
      - GLiClass-ModernBERT (default): ~50x faster than BART, 8K token context,
        uses efficient bi-encoder architecture. Model: knowledgator/gliclass-modern-base-v3.0
      - BART-MNLI (legacy): Slower but proven. Model: facebook/bart-large-mnli

    The backend is auto-detected from the model name.

    Parameters
    ----------
    settings : Settings
        Pipeline settings.
    model_name : str | None
        HuggingFace model for zero-shot classification.
    confidence_threshold : float | None
        Minimum confidence to assign a category (default 0.6).
    """

    def __init__(
        self,
        settings: Settings,
        model_name: str | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        self.settings = settings
        self.model_name = model_name or settings.classifier_model
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.classifier_confidence_threshold
        )
        self._pipeline = None
        self._gliclass_model = None
        self._is_gliclass = any(
            self.model_name.startswith(prefix) for prefix in _GLICLASS_PREFIXES
        )

    def _load_pipeline(self):
        """Lazy-load the classification pipeline (BART backend)."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "transformers is required for classification. "
                "Install with: pip install 'epstein-pipeline[classify]'"
            )

        logger.info("Loading BART classifier: %s", self.model_name)
        self._pipeline = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=-1,  # CPU by default; set device=0 for GPU
        )
        return self._pipeline

    def _load_gliclass(self):
        """Lazy-load the GLiClass model."""
        if self._gliclass_model is not None:
            return self._gliclass_model

        try:
            from gliner import GLiNER
        except ImportError:
            raise ImportError(
                "gliner is required for GLiClass classification. "
                "Install with: pip install gliner"
            )

        logger.info("Loading GLiClass classifier: %s", self.model_name)
        self._gliclass_model = GLiNER.from_pretrained(self.model_name)

        # Move to GPU if available
        try:
            import torch
            if torch.cuda.is_available():
                self._gliclass_model = self._gliclass_model.to("cuda")
                logger.info("GLiClass loaded on CUDA")
        except ImportError:
            pass

        return self._gliclass_model

    def _classify_gliclass(self, text: str, doc_id: str) -> ClassificationResult:
        """Classify using GLiClass-ModernBERT.

        GLiClass uses the same predict_entities API as GLiNER but for classification.
        Labels are passed as entity types and the model returns the best match.
        """
        model = self._load_gliclass()

        # GLiClass uses a different API than the HF pipeline
        # It takes text + candidate labels and returns scored labels
        try:
            # Use the model's built-in classification method
            entities = model.predict_entities(
                text[:4000],  # GLiClass handles up to 8K tokens, but truncate for speed
                CLASSIFICATION_LABELS,
                threshold=0.0,  # Get all scores
            )

            # Build scores dict
            scores: dict[str, float] = {}
            for entity in entities:
                label = entity.get("label", "")
                score = entity.get("score", 0.0)
                category = _LABEL_TO_CATEGORY.get(label, "other")
                if category not in scores or score > scores[category]:
                    scores[category] = round(score, 4)

            if entities:
                top = max(entities, key=lambda e: e.get("score", 0.0))
                top_label = top.get("label", "")
                top_score = top.get("score", 0.0)
                top_category = _LABEL_TO_CATEGORY.get(top_label, "other")

                if top_score < self.confidence_threshold:
                    top_category = "other"

                return ClassificationResult(
                    document_id=doc_id,
                    predicted_category=top_category,
                    confidence=round(top_score, 4),
                    all_scores=scores,
                )
        except Exception as exc:
            logger.warning("GLiClass classification failed, falling back to BART: %s", exc)
            # Fall back to BART pipeline
            self._is_gliclass = False
            return self._classify_bart(text, doc_id)

        return ClassificationResult(
            document_id=doc_id,
            predicted_category="other",
            confidence=0.0,
            all_scores={},
        )

    def _classify_bart(self, text: str, doc_id: str) -> ClassificationResult:
        """Classify using BART-MNLI (legacy)."""
        pipe = self._load_pipeline()
        result = pipe(
            text,
            candidate_labels=CLASSIFICATION_LABELS,
            multi_label=False,
        )

        scores: dict[str, float] = {}
        for label, score in zip(result["labels"], result["scores"]):
            category = _LABEL_TO_CATEGORY.get(label, "other")
            scores[category] = round(score, 4)

        top_label = result["labels"][0]
        top_score = result["scores"][0]
        top_category = _LABEL_TO_CATEGORY.get(top_label, "other")

        if top_score < self.confidence_threshold:
            top_category = "other"

        return ClassificationResult(
            document_id=doc_id,
            predicted_category=top_category,
            confidence=round(top_score, 4),
            all_scores=scores,
        )

    def classify(self, doc: Document) -> ClassificationResult:
        """Classify a single document."""
        # Build text from available fields
        parts = []
        if doc.title:
            parts.append(f"Title: {doc.title}")
        if doc.summary:
            parts.append(f"Summary: {doc.summary}")
        if doc.ocrText:
            # Use first 2000 chars of OCR text (GLiClass handles 8K tokens)
            parts.append(doc.ocrText[:2000])

        text = "\n".join(parts)
        if not text.strip():
            return ClassificationResult(
                document_id=doc.id,
                predicted_category="other",
                confidence=0.0,
                all_scores={},
            )

        if self._is_gliclass:
            return self._classify_gliclass(text, doc.id)
        else:
            return self._classify_bart(text, doc.id)

    def classify_batch(
        self,
        documents: list[Document],
        max_workers: int = 1,
    ) -> list[ClassificationResult]:
        """Classify multiple documents."""
        from rich.console import Console
        from epstein_pipeline.utils.progress import create_progress

        console = Console()
        results: list[ClassificationResult] = []
        backend = "GLiClass-ModernBERT" if self._is_gliclass else "BART-MNLI"
        console.print(f"  Classifying {len(documents)} documents with [bold]{backend}[/bold]")

        with create_progress() as progress:
            task = progress.add_task("Classifying", total=len(documents))
            for doc in documents:
                try:
                    result = self.classify(doc)
                    results.append(result)
                except Exception as exc:
                    logger.error("Classification failed for %s: %s", doc.id, exc)
                    results.append(
                        ClassificationResult(
                            document_id=doc.id,
                            predicted_category="other",
                            confidence=0.0,
                            all_scores={},
                        )
                    )
                progress.advance(task)

        return results
