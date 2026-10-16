from pathlib import Path

import pymupdf

#: Render resolution for page images; high enough for handwriting, small enough for the context window.
PAGE_DPI = 150


def pdf_page_count(pdf_path: Path) -> int:
    """The number of pages in ``pdf_path``."""
    with pymupdf.open(pdf_path) as doc:
        return len(doc)


def render_pdf_page(pdf_path: Path, page: int, dpi: int = PAGE_DPI) -> bytes:
    """Render 1-based ``page`` of ``pdf_path`` as PNG bytes.

    Raises:
        ValueError: If ``page`` is outside the document.
    """
    with pymupdf.open(pdf_path) as doc:
        return _page(doc, pdf_path, page).get_pixmap(dpi=dpi).tobytes("png")


def pdf_page_size(pdf_path: Path, page: int) -> tuple[float, float]:
    """The (width, height) in PDF points of 1-based ``page`` of ``pdf_path``.

    Uses the visible page area (CropBox, adjusted for rotation), which is what MarkUs displays,
    so pages of different sizes or orientations in one document each get their own size.

    Raises:
        ValueError: If ``page`` is outside the document.
    """
    with pymupdf.open(pdf_path) as doc:
        rect = _page(doc, pdf_path, page).rect
        return rect.width, rect.height


def _page(doc: pymupdf.Document, pdf_path: Path, page: int) -> pymupdf.Page:
    """1-based ``page`` of ``doc``, or a ValueError naming the file and its page count."""
    if not 1 <= page <= len(doc):
        raise ValueError(f"Page {page} is out of range; {Path(pdf_path).name} has {len(doc)} pages.")
    return doc[page - 1]


def extract_pdf_submission_text(pdf_path: Path) -> str:
    """The text of every page of ``pdf_path``, each under a ``--- Page n of N ---`` header."""
    with pymupdf.open(pdf_path) as doc:
        return "\n".join(f"--- Page {page.number + 1} of {len(doc)} ---\n{page.get_text()}" for page in doc)
