"""Media transcription with optional speaker diarization.

Two backends:
  1. faster-whisper (default) — fast, no diarization
  2. WhisperX + pyannote (--diarize) — word-level timestamps + speaker labels

Supports: .mp3, .mp4, .wav, .m4a, .avi, .wmv, .flac, .ogg, .webm, .mov
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from rich.console import Console

from epstein_pipeline.models.forensics import Transcript, TranscriptSegment

logger = logging.getLogger(__name__)
console = Console()

SUPPORTED_EXTENSIONS = {
    ".mp3", ".mp4", ".wav", ".m4a", ".avi", ".wmv",
    ".flac", ".ogg", ".webm", ".mov",
}


class MediaTranscriber:
    """Transcribe audio/video files with optional speaker diarization."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "auto",
        diarize: bool = False,
        hf_token: str | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.diarize = diarize
        self.hf_token = hf_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._model = None
        self._diarize_model = None

    def _resolve_device(self) -> tuple[str, str]:
        """Resolve device and compute type."""
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

        return device, compute

    def _ensure_model(self):
        """Lazy-load the transcription model."""
        if self._model is not None:
            return

        device, compute = self._resolve_device()

        if self.diarize:
            self._ensure_whisperx(device, compute)
        else:
            self._ensure_faster_whisper(device, compute)

    def _ensure_faster_whisper(self, device: str, compute: str):
        """Load faster-whisper model (no diarization).

        For large-v3-turbo on GPUs with ≤6GB VRAM, auto-selects INT8 quantization
        which cuts memory ~75% while maintaining near-identical accuracy.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is required. Install with: pip install faster-whisper"
            )

        # Auto-select INT8 for large models on ≤6GB VRAM GPUs
        model_id = self.model_size
        if device == "cuda" and "large" in self.model_size:
            try:
                import torch
                vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
                if vram_gb <= 8:
                    compute = "int8_float16"  # INT8 weights, FP16 activations — best for ≤8GB
                    console.print(f"  [dim]Auto-selected int8_float16 for {vram_gb:.0f}GB VRAM[/dim]")
            except Exception:
                pass

        console.print(
            f"  Loading faster-whisper [bold]{model_id}[/bold] on {device} ({compute})"
        )
        self._model = WhisperModel(model_id, device=device, compute_type=compute)

    def _ensure_whisperx(self, device: str, compute: str):
        """Load WhisperX model with diarization pipeline."""
        try:
            import whisperx
        except ImportError:
            raise ImportError(
                "WhisperX is required for diarization. Install with: "
                "pip install 'epstein-pipeline[transcription-diarize]'"
            )

        console.print(
            f"  Loading WhisperX [bold]{self.model_size}[/bold] on {device} ({compute})"
        )
        self._model = whisperx.load_model(
            self.model_size,
            device=device,
            compute_type=compute,
        )

        if self.hf_token:
            console.print("  Loading pyannote diarization pipeline...")
            self._diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=device,
            )
        else:
            console.print(
                "  [yellow]No HF_TOKEN — diarization disabled. "
                "Set HF_TOKEN env var for speaker identification.[/yellow]"
            )

    def transcribe_file(self, path: Path) -> Transcript:
        """Transcribe a single audio/video file."""
        self._ensure_model()

        if self.diarize:
            return self._transcribe_whisperx(path)
        else:
            return self._transcribe_faster_whisper(path)

    def _transcribe_faster_whisper(self, path: Path) -> Transcript:
        """Transcribe with faster-whisper (no diarization)."""
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

        elapsed = time.monotonic() - start
        logger.info("Transcribed %s in %.1fs", path.name, elapsed)

        return Transcript(
            source_path=str(path),
            document_id=f"tx-{path.stem}",
            text=" ".join(full_text_parts),
            language=info.language or "en",
            duration_seconds=info.duration or 0.0,
            segments=segments,
            diarized=False,
        )

    def _transcribe_whisperx(self, path: Path) -> Transcript:
        """Transcribe with WhisperX + optional speaker diarization."""
        import whisperx

        device, _ = self._resolve_device()
        start = time.monotonic()

        # Step 1: Transcribe with WhisperX
        audio = whisperx.load_audio(str(path))
        result = self._model.transcribe(audio, batch_size=16)
        language = result.get("language", "en")

        # Step 2: Align word-level timestamps
        try:
            align_model, align_metadata = whisperx.load_align_model(
                language_code=language, device=device
            )
            result = whisperx.align(
                result["segments"], align_model, align_metadata, audio, device,
                return_char_alignments=False,
            )
        except Exception as exc:
            logger.warning("Alignment failed for %s: %s — using raw timestamps", path.name, exc)

        # Step 3: Speaker diarization (if available)
        speakers_found: list[str] = []
        if self._diarize_model is not None:
            try:
                diarize_segments = self._diarize_model(
                    audio,
                    min_speakers=self.min_speakers,
                    max_speakers=self.max_speakers,
                )
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as exc:
                logger.warning("Diarization failed for %s: %s", path.name, exc)

        # Step 4: Build segments with speaker labels
        segments: list[TranscriptSegment] = []
        full_text_parts: list[str] = []

        for seg in result.get("segments", []):
            speaker = seg.get("speaker")
            text = seg.get("text", "").strip()
            if not text:
                continue

            if speaker and speaker not in speakers_found:
                speakers_found.append(speaker)

            segments.append(
                TranscriptSegment(
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=text,
                    speaker=speaker,
                    confidence=seg.get("confidence", 0.0) if "confidence" in seg else 0.0,
                )
            )
            full_text_parts.append(
                f"{speaker}: {text}" if speaker else text
            )

        elapsed = time.monotonic() - start
        duration = len(audio) / 16000  # WhisperX loads at 16kHz

        logger.info(
            "WhisperX transcribed %s in %.1fs (%.1fx realtime), %d speakers",
            path.name, elapsed, duration / max(elapsed, 0.1), len(speakers_found),
        )

        return Transcript(
            source_path=str(path),
            document_id=f"tx-{path.stem}",
            text=" ".join(full_text_parts),
            language=language,
            duration_seconds=duration,
            segments=segments,
            speakers=speakers_found,
            diarized=bool(self._diarize_model and speakers_found),
        )

    def transcribe_batch(
        self,
        paths: list[Path],
        output_dir: Path,
        max_workers: int = 1,
    ) -> list[Transcript]:
        """Transcribe multiple media files sequentially.

        Note: Whisper uses GPU memory, so sequential processing is recommended.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

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

                    # Save transcript as JSON
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

        if self.diarize:
            total_speakers = sum(len(t.speakers) for t in results)
            diarized_count = sum(1 for t in results if t.diarized)
            console.print(f"  Diarized: {diarized_count}/{len(results)} files, {total_speakers} unique speakers")

        return results
