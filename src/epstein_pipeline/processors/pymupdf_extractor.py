"""PyMuPDF-based PDF text, image, and redaction extraction.

PyMuPDF (fitz) is better than Docling at detecting invisible OCR text
layers (PDF rendering mode Tr=3) and can extract embedded images and
identify redaction regions (filled black rectangles).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PageText:
    """Text extracted from a single PDF page."""

    page_number: int
    text: str
    char_count: int = 0


@dataclass
class PageImage:
    """An image extracted from a single PDF page."""

    page_number: int
    image_index: int
    width: int
    height: int
    colorspace: str
    bpc: int  # bits per component
    image_bytes: bytes = field(repr=False)
    ext: str = "png"


@dataclass
class RedactionRegion:
    """A detected redaction region (filled black rectangle)."""

    page_number: int
    x0: float
    y0: float
    x1: float
    y1: float
    area: float
    text_under: str | None = None  # Text found under the redaction, if any
    classification: str = "unknown"  # "proper", "bad_overlay", "recoverable"


class PyMuPdfExtractor:
    """Extract text, images, and detect redactions from PDFs using PyMuPDF."""

    def __init__(self) -> None:
        try:
            import fitz  # noqa: F401
        except ImportError:
            raise ImportError("PyMuPDF is required. Install with: pip install pymupdf")

    def extract_text(self, path: Path) -> str:
        """Extract all text from a PDF, concatenated across pages."""
        import fitz

        doc = fitz.open(str(path))
        try:
            pages = []
            for page in doc:
                pages.append(page.get_text())
            return "\n\n".join(pages)
        finally:
            doc.close()

    def extract_pages(self, path: Path) -> list[PageText]:
        """Extract text from each page separately."""
        import fitz

        doc = fitz.open(str(path))
        try:
            results = []
            for i, page in enumerate(doc):
                text = page.get_text()
                results.append(
                    PageText(
                        page_number=i + 1,
                        text=text,
                        char_count=len(text),
                    )
                )
            return results
        finally:
            doc.close()

    def extract_images(self, path: Path) -> list[PageImage]:
        """Extract all embedded images from a PDF."""
        import fitz

        doc = fitz.open(str(path))
        try:
            results = []
            for page_num, page in enumerate(doc):
                image_list = page.get_images(full=True)
                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    try:
                        pix = fitz.Pixmap(doc, xref)
                        # Convert CMYK to RGB
                        if pix.n > 4:
                            pix = fitz.Pixmap(fitz.csRGB, pix)

                        results.append(
                            PageImage(
                                page_number=page_num + 1,
                                image_index=img_idx,
                                width=pix.width,
                                height=pix.height,
                                colorspace=str(pix.colorspace),
                                bpc=pix.n,
                                image_bytes=pix.tobytes("png"),
                                ext="png",
                            )
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to extract image %d on page %d of %s: %s",
                            img_idx,
                            page_num + 1,
                            path.name,
                            exc,
                        )
            return results
        finally:
            doc.close()

    def detect_redactions(self, path: Path) -> list[RedactionRegion]:
        """Detect redaction regions (filled black rectangles) in a PDF.

        For each detected redaction, attempts to extract any text that may
        be hidden underneath (bad overlay / recoverable redactions).
        """
        import fitz

        doc = fitz.open(str(path))
        try:
            results = []
            for page_num, page in enumerate(doc):
                # Get all drawing operations
                drawings = page.get_drawings()
                page_text_dict = page.get_text("dict")

                for drawing in drawings:
                    # Look for filled rectangles that are black/very dark
                    if drawing.get("fill") is None:
                        continue

                    fill_color = drawing.get("fill", (1, 1, 1))
                    # Check if fill is black or very dark
                    if isinstance(fill_color, (list, tuple)):
                        brightness = sum(fill_color[:3]) / max(len(fill_color[:3]), 1)
                        if brightness > 0.1:  # Not dark enough to be a redaction
                            continue
                    else:
                        continue

                    rect = drawing.get("rect")
                    if rect is None:
                        continue

                    x0, y0, x1, y1 = rect
                    area = abs(x1 - x0) * abs(y1 - y0)

                    # Skip tiny regions (likely dots or lines)
                    if area < 100:
                        continue

                    # Try to find text underneath this rectangle
                    text_under = self._get_text_under_rect(page, page_text_dict, x0, y0, x1, y1)

                    if text_under:
                        classification = "recoverable"
                    else:
                        classification = "proper"

                    results.append(
                        RedactionRegion(
                            page_number=page_num + 1,
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                            area=area,
                            text_under=text_under,
                            classification=classification,
                        )
                    )

            return results
        finally:
            doc.close()

    @staticmethod
    def _get_text_under_rect(
        page,
        text_dict: dict,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> str | None:
        """Try to extract text that falls within a rectangle region."""
        import fitz

        rect = fitz.Rect(x0, y0, x1, y1)
        text_parts = []

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_rect = fitz.Rect(span["bbox"])
                    if rect.intersects(span_rect):
                        text = span.get("text", "").strip()
                        if text:
                            text_parts.append(text)

        combined = " ".join(text_parts).strip()
        return combined if combined else None
