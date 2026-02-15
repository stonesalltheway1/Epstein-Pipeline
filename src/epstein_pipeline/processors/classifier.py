"""Zero-shot document classification using transformer models.

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


@dataclass
class ClassificationResult:
    """Result of classifying a single document."""

    document_id: str
    predicted_category: DocumentCategory
    confidence: float
    all_scores: dict[str, float]


class DocumentClassifier:
    """Classify documents into categories using zero-shot classification.

    Uses ``facebook/bart-large-mnli`` by default for zero-shot classification.
    Only assigns a category if confidence exceeds the threshold.

    Parameters
    ----------
    settings : Settings
        Pipeline settings.
    model_name : str | None
        HuggingFace model for zero-shot classification.
    confidence_threshold : float
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

    def _load_pipeline(self):
        """Lazy-load the classification pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "transformers is required for classification. "
                "Install with: pip install 'epstein-pipeline[classify]'"
            )

        logger.info("Loading classifier model: %s", self.model_name)
        self._pipeline = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=-1,  # CPU by default; set device=0 for GPU
        )
        return self._pipeline

    def classify(self, doc: Document) -> ClassificationResult:
        """Classify a single document."""
        # Build text from available fields
        parts = []
        if doc.title:
            parts.append(f"Title: {doc.title}")
        if doc.summary:
            parts.append(f"Summary: {doc.summary}")
        if doc.ocrText:
            # Use first 1000 chars of OCR text for classification
            parts.append(doc.ocrText[:1000])

        text = "\n".join(parts)
        if not text.strip():
            return ClassificationResult(
                document_id=doc.id,
                predicted_category="other",
                confidence=0.0,
                all_scores={},
            )

        pipe = self._load_pipeline()
        result = pipe(
            text,
            candidate_labels=CLASSIFICATION_LABELS,
            multi_label=False,
        )

        # Parse results
        scores: dict[str, float] = {}
        for label, score in zip(result["labels"], result["scores"]):
            category = _LABEL_TO_CATEGORY.get(label, "other")
            scores[category] = round(score, 4)

        # Get top prediction
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        top_category = _LABEL_TO_CATEGORY.get(top_label, "other")

        # Only assign if above threshold
        if top_score < self.confidence_threshold:
            top_category = "other"

        return ClassificationResult(
            document_id=doc.id,
            predicted_category=top_category,
            confidence=round(top_score, 4),
            all_scores=scores,
        )

    def classify_batch(
        self,
        documents: list[Document],
        max_workers: int = 1,
    ) -> list[ClassificationResult]:
        """Classify multiple documents.

        Note: The transformers pipeline handles batching internally,
        so we process sequentially by default.
        """
        results: list[ClassificationResult] = []

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

        return results
