"""PLIST / Apple Mail forensics processor.

Some DOJ-released PDFs contain embedded Apple Mail PLIST metadata from
the original email conversion process.  This processor detects and
extracts that metadata, which can reveal sender/recipient information,
timestamps, and mail headers that may not be visible in the PDF text.
"""

from __future__ import annotations

import logging
import plistlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

# PLIST magic bytes / markers that appear in PDF streams
_PLIST_MARKERS = [
    b"<?xml version=",
    b"<plist",
    b"bplist00",  # binary plist
]

_PLIST_XML_PATTERN = re.compile(
    rb"<\?xml\s+version=[^?]+\?>\s*.*?<plist[^>]*>.*?</plist>",
    re.DOTALL,
)


@dataclass
class PlistMetadata:
    """Metadata extracted from an embedded PLIST."""

    document_id: str
    plist_type: str  # "xml", "binary"
    sender: str | None = None
    recipients: list[str] = field(default_factory=list)
    subject: str | None = None
    date: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    raw_keys: list[str] = field(default_factory=list)


class PlistForensicsProcessor:
    """Detect and extract Apple Mail PLIST metadata from PDFs."""

    def detect_plist(self, path: Path) -> bool:
        """Check if a PDF contains embedded PLIST data."""
        try:
            raw = path.read_bytes()
            return any(marker in raw for marker in _PLIST_MARKERS)
        except Exception:
            return False

    def extract_plist(self, path: Path) -> list[PlistMetadata]:
        """Extract all PLIST metadata from a PDF file."""
        try:
            raw = path.read_bytes()
        except Exception as exc:
            logger.error("Failed to read %s: %s", path, exc)
            return []

        doc_id = path.stem
        results: list[PlistMetadata] = []

        # Try XML PLISTs
        for match in _PLIST_XML_PATTERN.finditer(raw):
            try:
                plist_data = plistlib.loads(match.group())
                metadata = self._parse_plist_dict(doc_id, plist_data, "xml")
                if metadata:
                    results.append(metadata)
            except Exception as exc:
                logger.debug("Failed to parse XML PLIST in %s: %s", path.name, exc)

        # Try binary PLISTs
        bplist_starts = [i for i in range(len(raw)) if raw[i : i + 8] == b"bplist00"]
        for start in bplist_starts:
            # Binary PLISTs don't have a clear end marker, try chunks
            for end in range(start + 100, min(start + 100_000, len(raw)), 1000):
                try:
                    plist_data = plistlib.loads(raw[start:end])
                    metadata = self._parse_plist_dict(doc_id, plist_data, "binary")
                    if metadata:
                        results.append(metadata)
                    break  # Found valid end
                except Exception:
                    continue

        return results

    def _parse_plist_dict(
        self, doc_id: str, data: dict | list | str, plist_type: str
    ) -> PlistMetadata | None:
        """Parse a PLIST dictionary into metadata."""
        if not isinstance(data, dict):
            return None

        # Common Apple Mail PLIST keys
        sender = (
            data.get("sender")
            or data.get("from")
            or data.get("X-Sender")
            or data.get("kMDItemAuthorAddresses", [None])[0]
            if isinstance(data.get("kMDItemAuthorAddresses"), list)
            else data.get("kMDItemAuthorAddresses")
        )

        recipients = []
        for key in ("to", "recipients", "X-To", "kMDItemRecipientAddresses"):
            val = data.get(key)
            if isinstance(val, list):
                recipients.extend(str(v) for v in val)
            elif isinstance(val, str):
                recipients.append(val)

        subject = data.get("subject") or data.get("Subject") or data.get("kMDItemSubject")
        date = data.get("date") or data.get("Date") or data.get("kMDItemContentCreationDate")

        if date and not isinstance(date, str):
            date = str(date)

        # Extract all string headers
        headers = {}
        for key, val in data.items():
            if isinstance(val, str) and len(val) < 1000:
                headers[key] = val

        metadata = PlistMetadata(
            document_id=doc_id,
            plist_type=plist_type,
            sender=str(sender) if sender else None,
            recipients=recipients,
            subject=str(subject) if subject else None,
            date=date,
            headers=headers,
            raw_keys=list(data.keys()),
        )

        # Only return if we got something useful
        if metadata.sender or metadata.recipients or metadata.subject:
            return metadata
        return None

    def process_batch(
        self,
        paths: list[Path],
        output_dir: Path,
    ) -> list[PlistMetadata]:
        """Scan multiple PDFs for PLIST metadata."""
        import json

        output_dir.mkdir(parents=True, exist_ok=True)
        all_results: list[PlistMetadata] = []
        detected = 0

        from epstein_pipeline.utils.progress import create_progress

        with create_progress() as progress:
            task = progress.add_task("Scanning for PLISTs", total=len(paths))
            for path in paths:
                if self.detect_plist(path):
                    detected += 1
                    results = self.extract_plist(path)
                    all_results.extend(results)
                progress.advance(task)

        # Write results
        if all_results:
            out_path = output_dir / "plist_forensics.json"
            data = [
                {
                    "document_id": m.document_id,
                    "plist_type": m.plist_type,
                    "sender": m.sender,
                    "recipients": m.recipients,
                    "subject": m.subject,
                    "date": m.date,
                    "headers": m.headers,
                    "raw_keys": m.raw_keys,
                }
                for m in all_results
            ]
            out_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

        console.print(f"\n  Scanned {len(paths)} files, {detected} contain PLIST data")
        console.print(f"  Extracted {len(all_results)} PLIST metadata records")

        return all_results
