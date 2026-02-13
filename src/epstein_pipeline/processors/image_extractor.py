"""Extract images from PDFs and optionally describe them with a vision model.

Uses PyMuPDF for extraction and supports Ollama (llava) or OpenAI
(gpt-4o-mini) for AI-generated image descriptions.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from pathlib import Path

import httpx
from rich.console import Console

from epstein_pipeline.models.forensics import ExtractedImage

logger = logging.getLogger(__name__)
console = Console()


class ImageExtractor:
    """Extract and optionally describe images from PDF files."""

    def __init__(
        self,
        vision_model: str = "llava",
        vision_provider: str = "ollama",
    ) -> None:
        self.vision_model = vision_model
        self.vision_provider = vision_provider

        try:
            from epstein_pipeline.processors.pymupdf_extractor import PyMuPdfExtractor

            self._extractor = PyMuPdfExtractor()
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for image extraction. Install with: pip install pymupdf"
            )

    def extract_images(self, path: Path) -> list[ExtractedImage]:
        """Extract all images from a PDF and return metadata."""
        doc_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        doc_id = f"img-{doc_hash}"

        raw_images = self._extractor.extract_images(path)

        results = []
        for img in raw_images:
            results.append(
                ExtractedImage(
                    document_id=doc_id,
                    page_number=img.page_number,
                    image_index=img.image_index,
                    width=img.width,
                    height=img.height,
                    format=img.ext,
                    size_bytes=len(img.image_bytes),
                )
            )

        return results

    def extract_and_save(
        self,
        path: Path,
        output_dir: Path,
        describe: bool = False,
    ) -> list[ExtractedImage]:
        """Extract images, save to disk, and optionally describe them."""
        doc_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        doc_id = f"img-{doc_hash}"
        img_dir = output_dir / doc_id
        img_dir.mkdir(parents=True, exist_ok=True)

        raw_images = self._extractor.extract_images(path)
        results = []

        for img in raw_images:
            filename = f"page{img.page_number:04d}_img{img.image_index:03d}.{img.ext}"
            img_path = img_dir / filename
            img_path.write_bytes(img.image_bytes)

            description = None
            if describe:
                try:
                    description = self.describe_image(img.image_bytes)
                except Exception as exc:
                    logger.warning("Image description failed: %s", exc)

            results.append(
                ExtractedImage(
                    document_id=doc_id,
                    page_number=img.page_number,
                    image_index=img.image_index,
                    width=img.width,
                    height=img.height,
                    format=img.ext,
                    file_path=str(img_path),
                    description=description,
                    size_bytes=len(img.image_bytes),
                )
            )

        return results

    def describe_image(self, image_bytes: bytes) -> str | None:
        """Send an image to a vision model for description."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = (
            "Describe this image from the Epstein case files. "
            "Be factual and concise. Note any visible text, people, "
            "locations, or document types."
        )

        if self.vision_provider == "ollama":
            return self._describe_ollama(b64, prompt)
        elif self.vision_provider == "openai":
            return self._describe_openai(b64, prompt)
        return None

    def _describe_ollama(self, b64_image: str, prompt: str) -> str | None:
        """Use Ollama's vision model for image description."""
        try:
            resp = httpx.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.vision_model,
                    "prompt": prompt,
                    "images": [b64_image],
                    "stream": False,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as exc:
            logger.warning("Ollama vision failed: %s", exc)
            return None

    def _describe_openai(self, b64_image: str, prompt: str) -> str | None:
        """Use OpenAI's vision model for image description."""
        try:
            import openai

            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_image}",
                                    "detail": "low",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("OpenAI vision failed: %s", exc)
            return None

    def process_batch(
        self,
        paths: list[Path],
        output_dir: Path,
        describe: bool = False,
        max_workers: int = 1,
    ) -> list[ExtractedImage]:
        """Extract images from multiple PDFs."""
        output_dir.mkdir(parents=True, exist_ok=True)
        all_images: list[ExtractedImage] = []

        from epstein_pipeline.utils.progress import create_progress

        with create_progress() as progress:
            task = progress.add_task("Extracting images", total=len(paths))
            for path in paths:
                try:
                    images = self.extract_and_save(path, output_dir, describe=describe)
                    all_images.extend(images)
                except Exception as exc:
                    logger.error("Image extraction failed for %s: %s", path, exc)
                progress.advance(task)

        console.print(
            f"\n  [green]Extracted {len(all_images)} images from {len(paths)} files[/green]"
        )
        return all_images
