"""Media transcription using faster-whisper.

Supports audio and video files: .mp3, .mp4, .wav, .m4a, .avi, .wmv, .flac.
Uses faster-whisper for GPU-accelerated transcription with the large-v3 model.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from rich.console import Console

from epstein_pipeline.models.forensics import Transcript, TranscriptSegment

logger = logging.getLogger(__name__)
console = Console()

SUPPORTED_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a", ".avi", ".wmv", ".flac", ".ogg", ".webm"}


class MediaTranscriber:
    """Transcribe audio/video files using faster-whisper."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "auto",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _ensure_model(self):
        """Lazy-load the whisper model."""
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is required for transcription. "
                "Install with: pip install faster-whisper"
            )

        # Auto-detect device
        device = self.device
        compute = self.compute_type

        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"

        console.print(
            f"  Loading whisper model [bold]{self.model_size}[/bold] on {device} ({compute})"
        )
        self._model = WhisperModel(self.model_size, device=device, compute_type=compute)

    def transcribe_file(self, path: Path) -> Transcript:
        """Transcribe a single audio/video file."""
        self._ensure_model()

        start = time.monotonic()
        segments_iter, info = self._model.transcribe(
            str(path),
            beam_size=5,
            vad_filter=True,
        )

        segments = []
        full_text_parts = []

        for segment in segments_iter:
            segments.append(
                TranscriptSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                )
            )
            full_text_parts.append(segment.text.strip())

        time.monotonic() - start

        return Transcript(
            source_path=str(path),
            document_id=f"tx-{path.stem}",
            text=" ".join(full_text_parts),
            language=info.language or "en",
            duration_seconds=info.duration or 0.0,
            segments=segments,
        )

    def transcribe_batch(
        self,
        paths: list[Path],
        output_dir: Path,
        max_workers: int = 1,
    ) -> list[Transcript]:
        """Transcribe multiple media files.

        Note: Whisper uses GPU memory, so max_workers > 1 may cause OOM.
        Sequential processing is recommended for GPU transcription.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Filter to supported extensions
        valid_paths = [p for p in paths if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if len(valid_paths) < len(paths):
            skipped = len(paths) - len(valid_paths)
            console.print(f"  [yellow]Skipping {skipped} unsupported files[/yellow]")

        results: list[Transcript] = []

        from epstein_pipeline.utils.progress import create_progress

        with create_progress() as progress:
            task = progress.add_task("Transcribing", total=len(valid_paths))
            for path in valid_paths:
                progress.update(task, description=f"Transcribing: {path.name[:40]}")
                try:
                    transcript = self.transcribe_file(path)
                    results.append(transcript)

                    # Save transcript
                    out_path = output_dir / f"{path.stem}.json"
                    out_path.write_text(
                        transcript.model_dump_json(indent=2),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    logger.error("Transcription failed for %s: %s", path, exc)
                progress.advance(task)

        console.print(f"\n  [green]Transcribed {len(results)} files[/green]")
        total_duration = sum(t.duration_seconds for t in results)
        console.print(f"  Total audio duration: {total_duration / 3600:.1f} hours")

        return results
