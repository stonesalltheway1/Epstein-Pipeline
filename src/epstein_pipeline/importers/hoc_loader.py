"""House Oversight Committee Concordance/Relativity load file importer.

Parses the .dat (metadata) and .opt (page-to-image mapping) files from the
Kaggle `jazivxt/the-epstein-files` dataset. Groups page images into logical
documents, extracts metadata, and assembles multi-page PDFs.

The .dat format uses þ (thorn, U+00FE) as a field delimiter/wrapper.
Each field value is enclosed in þ...þ, so empty values appear as þþ.

The .opt format is comma-delimited:
  BatesID, Volume, ImagePath, DocStart(Y/blank), unused, unused, PageCount
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HocPage:
    """A single page image from the HOC release."""
    bates_id: str               # e.g. "HOUSE_OVERSIGHT_010477"
    image_path: str             # relative path from .opt
    is_doc_start: bool          # True if this page starts a new document
    page_count: int | None      # only set on doc-start pages


@dataclass
class HocDocument:
    """A logical document composed of one or more pages."""
    bates_begin: str
    bates_end: str
    bates_begin_num: int        # numeric portion for ID generation
    pages: list[HocPage] = field(default_factory=list)

    # Metadata from .dat file
    page_count: int = 0
    author: str = ""
    custodian: str = ""
    date_created: str = ""
    date_modified: str = ""
    date_received: str = ""
    date_sent: str = ""
    doc_extension: str = ""
    email_from: str = ""
    email_to: str = ""
    email_cc: str = ""
    email_bcc: str = ""
    email_subject: str = ""
    original_filename: str = ""
    file_size: int = 0
    original_folder: str = ""
    md5_hash: str = ""
    parent_doc_id: str = ""
    doc_title: str = ""
    text_link: str = ""
    native_link: str = ""

    @property
    def doc_id(self) -> str:
        """Generate the kaggle-ho-XXXXXX ID matching our existing format."""
        return f"kaggle-ho-{self.bates_begin_num:06d}"

    @property
    def best_title(self) -> str:
        """Return the best available title."""
        if self.doc_title:
            return self.doc_title
        if self.email_subject:
            return self.email_subject
        if self.original_filename:
            return self.original_filename
        return f"HOUSE_OVERSIGHT_{self.bates_begin_num:06d}"

    @property
    def best_date(self) -> str | None:
        """Return the best available date in ISO format."""
        for d in [self.date_sent, self.date_created, self.date_modified, self.date_received]:
            if d:
                return self._parse_date(d)
        return None

    @staticmethod
    def _parse_date(date_str: str) -> str | None:
        """Convert MM/DD/YYYY to YYYY-MM-DD."""
        m = re.match(r"(\d{2})/(\d{2})/(\d{4})", date_str)
        if m:
            return f"{m.group(3)}-{m.group(1)}-{m.group(2)}"
        return None

    @property
    def category(self) -> str:
        """Infer document category from extension and metadata."""
        ext = self.doc_extension.lower()
        if ext in ("msg", "eml"):
            return "correspondence"
        if ext in ("xls", "xlsx", "csv"):
            return "financial"
        if ext == "pdf" and any(kw in self.original_filename.lower() for kw in
                                ["invoice", "statement", "receipt", "tax", "bank"]):
            return "financial"
        if self.email_from or self.email_to:
            return "correspondence"
        return "other"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class HocLoader:
    """Parse Concordance .dat + .opt files into HocDocument objects."""

    def __init__(self, data_dir: Path, images_dirs: list[Path] | None = None):
        """
        Parameters
        ----------
        data_dir : Path
            Directory containing HOUSE_OVERSIGHT_009.dat and .opt files.
        images_dirs : list[Path], optional
            Directories containing the extracted IMAGES subdirectories.
        """
        self.data_dir = Path(data_dir)
        self.images_dirs = [Path(d) for d in (images_dirs or [])]

    def parse_opt(self, opt_path: Path | None = None) -> list[HocPage]:
        """Parse the .opt file to get page-to-image mappings."""
        if opt_path is None:
            opt_path = self.data_dir / "HOUSE_OVERSIGHT_009.opt"

        pages: list[HocPage] = []
        with open(opt_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4:
                    continue
                bates_id = row[0].strip()
                image_path = row[2].strip()
                is_doc_start = row[3].strip().upper() == "Y"
                page_count = int(row[6]) if len(row) > 6 and row[6].strip() else None

                pages.append(HocPage(
                    bates_id=bates_id,
                    image_path=image_path,
                    is_doc_start=is_doc_start,
                    page_count=page_count,
                ))

        console.print(f"  Parsed {len(pages):,} pages from .opt")
        return pages

    def parse_dat(self, dat_path: Path | None = None) -> dict[str, dict]:
        """Parse the .dat file to get document metadata.

        Returns a dict keyed by Bates Begin ID.
        """
        if dat_path is None:
            dat_path = self.data_dir / "HOUSE_OVERSIGHT_009.dat"

        # Read the file and parse the thorn-delimited format
        with open(dat_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()

        if not lines:
            return {}

        # Parse header: þField1þþField2þþField3þ
        # Split on þ gives: ['', 'Field1', '', 'Field2', '', 'Field3', '']
        # Real field names are at ODD indices: 1, 3, 5, ...
        header_parts = lines[0].strip().split("þ")
        field_names = [header_parts[i] for i in range(1, len(header_parts), 2)]

        documents: dict[str, dict] = {}
        for line in lines[1:]:
            parts = line.strip().split("þ")
            # Values are also at odd indices
            row_values = [parts[i] if i < len(parts) else ""
                          for i in range(1, max(len(parts), len(header_parts)), 2)]

            if not row_values:
                continue

            doc = {}
            for i, name in enumerate(field_names):
                if i < len(row_values):
                    doc[name] = row_values[i]
                else:
                    doc[name] = ""

            bates_begin = doc.get("Bates Begin", "")
            if bates_begin:
                documents[bates_begin] = doc

        console.print(f"  Parsed {len(documents):,} documents from .dat")
        return documents

    def group_pages_into_documents(
        self, pages: list[HocPage], metadata: dict[str, dict]
    ) -> list[HocDocument]:
        """Group pages into documents using doc-start markers from .opt
        and metadata from .dat."""

        documents: list[HocDocument] = []
        current_doc: HocDocument | None = None

        for page in pages:
            if page.is_doc_start:
                # Save previous document
                if current_doc is not None:
                    documents.append(current_doc)

                # Extract numeric Bates ID
                m = re.search(r"(\d+)$", page.bates_id)
                bates_num = int(m.group(1)) if m else 0

                current_doc = HocDocument(
                    bates_begin=page.bates_id,
                    bates_end=page.bates_id,
                    bates_begin_num=bates_num,
                    page_count=page.page_count or 1,
                )

                # Enrich with .dat metadata if available
                meta = metadata.get(page.bates_id, {})
                if meta:
                    current_doc.bates_end = meta.get("Bates End", page.bates_id)
                    current_doc.author = meta.get("Author", "")
                    current_doc.custodian = meta.get("Custodian/Source", "")
                    current_doc.date_created = meta.get("Date Created", "")
                    current_doc.date_modified = meta.get("Date Last Modified", "")
                    current_doc.date_received = meta.get("Date Received", "")
                    current_doc.date_sent = meta.get("Date Sent", "")
                    current_doc.doc_extension = meta.get("Document Extension", "")
                    current_doc.email_from = meta.get("Email From", "")
                    current_doc.email_to = meta.get("Email To", "")
                    current_doc.email_cc = meta.get("Email CC", "")
                    current_doc.email_bcc = meta.get("Email BCC", "")
                    current_doc.email_subject = meta.get("Email Subject/Title", "")
                    current_doc.original_filename = meta.get("Original Filename", "")
                    current_doc.original_folder = meta.get("Original Folder Path", "")
                    current_doc.md5_hash = meta.get("MD5 Hash", "")
                    current_doc.parent_doc_id = meta.get("Parent Document ID", "")
                    current_doc.doc_title = meta.get("Document Title", "")
                    current_doc.text_link = meta.get("Text Link", "")
                    current_doc.native_link = meta.get("Native Link", "")
                    try:
                        current_doc.file_size = int(meta.get("File Size", "0") or "0")
                    except ValueError:
                        current_doc.file_size = 0
                    try:
                        current_doc.page_count = int(meta.get("Pages", "0") or "0")
                    except ValueError:
                        pass

            if current_doc is not None:
                current_doc.pages.append(page)
                current_doc.bates_end = page.bates_id

        # Don't forget the last document
        if current_doc is not None:
            documents.append(current_doc)

        console.print(f"  Grouped into {len(documents):,} documents")
        return documents

    def resolve_image_path(self, relative_path: str) -> Path | None:
        """Resolve a relative image path from the .opt file to an actual file.

        The .opt paths look like: \\HOUSE_OVERSIGHT_009\\IMAGES\\001\\HOUSE_OVERSIGHT_010477.jpg
        The actual extracted paths vary by archive structure.
        """
        # Normalize path separators
        clean = relative_path.replace("\\", "/").lstrip("/")

        # Try each images directory
        for images_dir in self.images_dirs:
            # Direct match
            candidate = images_dir / clean
            if candidate.exists():
                return candidate

            # Try without the volume prefix (HOUSE_OVERSIGHT_009/)
            parts = clean.split("/")
            if len(parts) > 1:
                without_volume = "/".join(parts[1:])
                candidate = images_dir / without_volume
                if candidate.exists():
                    return candidate

            # Try just the filename in the IMAGES subdirectory structure
            filename = parts[-1]
            # Search for the file
            matches = list(images_dir.rglob(filename))
            if matches:
                return matches[0]

        return None

    def load(self) -> list[HocDocument]:
        """Full parse: .opt + .dat -> grouped documents."""
        console.print("[bold]Parsing House Oversight Concordance files[/bold]")

        pages = self.parse_opt()
        metadata = self.parse_dat()
        documents = self.group_pages_into_documents(pages, metadata)

        # Stats
        total_pages = sum(len(d.pages) for d in documents)
        with_metadata = sum(1 for d in documents if d.original_filename or d.doc_title)
        with_dates = sum(1 for d in documents if d.best_date)
        with_email = sum(1 for d in documents if d.email_from or d.email_to)

        console.print(f"\n  [green]Documents:[/green] {len(documents):,}")
        console.print(f"  [green]Total pages:[/green] {total_pages:,}")
        console.print(f"  [green]With metadata:[/green] {with_metadata:,}")
        console.print(f"  [green]With dates:[/green] {with_dates:,}")
        console.print(f"  [green]With email fields:[/green] {with_email:,}")

        return documents


# ---------------------------------------------------------------------------
# PDF Assembler
# ---------------------------------------------------------------------------

def assemble_pdf(
    doc: HocDocument,
    loader: HocLoader,
    output_dir: Path,
) -> Path | None:
    """Assemble a document's page images into a single PDF.

    Uses PyMuPDF (fitz) for fast image-to-PDF conversion.
    Returns the output PDF path, or None if no images found.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        console.print("[red]PyMuPDF required: pip install pymupdf[/red]")
        return None

    resolved_pages: list[Path] = []
    for page in doc.pages:
        img_path = loader.resolve_image_path(page.image_path)
        if img_path and img_path.exists():
            resolved_pages.append(img_path)

    if not resolved_pages:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{doc.doc_id}.pdf"

    if pdf_path.exists():
        return pdf_path  # Already assembled

    pdf = fitz.open()
    for img_path in resolved_pages:
        try:
            img = fitz.open(str(img_path))
            # Convert image to PDF page
            pdf_bytes = img.convert_to_pdf()
            img_pdf = fitz.open("pdf", pdf_bytes)
            pdf.insert_pdf(img_pdf)
            img.close()
            img_pdf.close()
        except Exception:
            continue

    if pdf.page_count > 0:
        pdf.save(str(pdf_path))
        pdf.close()
        return pdf_path
    else:
        pdf.close()
        return None


# ---------------------------------------------------------------------------
# CLI entry point (for testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "E:/Epstein-Pipeline/ingest/kaggle-hoc/DATA-20251116T222054Z-1-001/DATA"
    )

    images_dirs = [
        Path("E:/Epstein-Pipeline/ingest/kaggle-hoc/IMAGES-20251116T222057Z-1-001"),
        Path("E:/Epstein-Pipeline/ingest/kaggle-hoc/IMAGES-20251116T222057Z-1-002"),
    ]

    loader = HocLoader(data_dir, images_dirs)
    documents = loader.load()

    # Print sample documents
    console.print("\n[bold]Sample documents:[/bold]")
    for doc in documents[:10]:
        console.print(
            f"  {doc.doc_id} | {len(doc.pages)}p | "
            f"{doc.best_title[:60]} | {doc.best_date or 'no date'} | "
            f"{doc.category}"
        )
