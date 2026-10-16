"""Place model comments on a PDF page as MarkUs PDF annotations.

The model says *what* a comment is about (a quote, a figure, a part of a figure) and
PyMuPDF finds *where* that is. Placement falls back in layers, so a comment always gets
a location even on scanned or hand-drawn pages:

1. ``quote``       — the quote, found in the page's text layer
   ``figure_part`` — a part of a vector plot (title, axes, legend, panel, a bar)
   ``figure``      — a whole figure (vector drawing or embedded image)
2. ``ocr_quote``   — the quote, found after running OCR on the page
3. ``model_box``   — the model's own box, kept inside the named figure if there is one
4. ``top_strip``   — a fixed strip at the top of the page
"""

import difflib
import functools
import json
from pathlib import Path
from typing import Optional

import pymupdf

from .pdf_pages import _page

#: MarkUs stores PDF annotation coordinates as a fraction of the page multiplied by this.
MARKUS_COORDINATE_SCALE = 100_000

#: Where a comment goes when nothing better can be found, in MarkUs coordinates.
TOP_STRIP = {"x1": 5_000, "y1": 1_000, "x2": 95_000, "y2": 5_000}

#: Drawings and images smaller than this (in points, either side) are not treated as figures.
MIN_FIGURE_SIZE = 100

#: An image covering more than this share of the page is a scanned page, not a figure.
MAX_FIGURE_PAGE_SHARE = 0.8

#: Axis labels and titles can sit outside the drawing cluster; widen the figure by this (points).
FIGURE_MARGIN = (-40, -30, 10, 30)

#: Axis labels and titles further than this (points) from the plotting area belong to something else.
LABEL_REACH = 45

#: Legends can sit further from the plotting area than axis labels.
LEGEND_REACH = 150

#: Quote matches closer than this (points) vertically are joined, for quotes that wrap lines.
LINE_GAP = 20

#: OCR misreads characters (e.g. "Q1a" as "Qia"); accept OCR text this similar to the quote.
OCR_MIN_SIMILARITY = 0.8

#: Resolution for OCR of pages without a usable text layer.
OCR_DPI = 300

FIGURE_PARTS = ("none", "whole", "title", "x_axis", "y_axis", "legend", "panel", "bar")

_WHITE = (1.0, 1.0, 1.0)


class PageLocator:
    """Finds where comments belong on one page of a PDF.

    Use as a context manager; figures and OCR text are computed once per page, on first use.
    """

    _ocr_unavailable = False  # set once Tesseract fails, so OCR is not retried for every page

    def __init__(self, pdf_path: Path, page: int) -> None:
        self.pdf_path = Path(pdf_path)
        self.page_number = page
        self._doc: Optional[pymupdf.Document] = None
        self._page: Optional[pymupdf.Page] = None
        self._figures: Optional[list[pymupdf.Rect]] = None
        self._ocr_textpage = None

    def __enter__(self) -> "PageLocator":
        self._doc = pymupdf.open(self.pdf_path)
        self._page = _page(self._doc, self.pdf_path, self.page_number)
        return self

    def __exit__(self, *exc) -> None:
        self._doc.close()

    def place(self, annotation: dict) -> tuple[dict, str]:
        """MarkUs ``x1, y1, x2, y2`` for ``annotation`` and the layer that placed it.

        ``annotation`` may carry ``quote``, ``figure``, ``figure_part``, ``bar``, and
        ``x1, y1, x2, y2`` already on MarkUs's 0 to MARKUS_COORDINATE_SCALE scale.
        """
        quote = (annotation.get("quote") or "").strip() or None
        part = None if annotation.get("figure_part") == "none" else annotation.get("figure_part")
        figure = self._figure(annotation.get("figure"))
        search_area = figure + FIGURE_MARGIN if figure is not None else None

        if quote:
            rect = self._find_quote(quote, search_area)
            if rect is not None:
                return self._to_markus(rect), "quote"
        if figure is not None:
            if part in (None, "whole"):
                return self._to_markus(figure), "figure"
            rect = self._figure_part(figure, part, annotation.get("bar"))
            if rect is not None:
                return self._to_markus(rect), "figure_part"
        if quote:
            textpage = self._ocr()
            if textpage is not None:
                rect = self._find_quote(quote, search_area, textpage) or self._find_similar(
                    quote, search_area, textpage
                )
                if rect is not None:
                    return self._to_markus(rect), "ocr_quote"

        box = self._model_box(annotation)
        if figure is not None:
            # A part of a figure that cannot be found inside (e.g. a photographed plot).
            if box is not None and (box & figure).get_area() > 0:
                return self._to_markus(box & figure), "model_box"
            return self._to_markus(figure), "figure"
        if box is not None:
            return self._to_markus(box), "model_box"
        return dict(TOP_STRIP), "top_strip"

    def _figure(self, number) -> Optional[pymupdf.Rect]:
        """The ``number``-th figure on the page, counting from the top; None if absent."""
        if not isinstance(number, int) or isinstance(number, bool):
            return None
        if self._figures is None:
            page_area = self._page.rect.get_area()
            drawings = [r for r in self._page.cluster_drawings() if _is_figure_sized(r)]
            images = [
                pymupdf.Rect(r)
                for image in self._page.get_images()
                for r in self._page.get_image_rects(image[0])
                if _is_figure_sized(r) and r.get_area() <= MAX_FIGURE_PAGE_SHARE * page_area
            ]
            self._figures = sorted(drawings + images, key=lambda r: (r.y0, r.x0))
        return self._figures[number - 1] if 1 <= number <= len(self._figures) else None

    def _figure_part(self, figure: pymupdf.Rect, part: str, bar) -> Optional[pymupdf.Rect]:
        """A part of a vector plot, located from its drawings and text; None if it has no such part."""
        area = figure + FIGURE_MARGIN
        drawings = [d for d in self._page.get_drawings() if area.contains(d["rect"])]
        filled = [d for d in drawings if d.get("fill") and tuple(d["fill"]) != _WHITE]
        if not filled:
            return None  # an image, or a plot with no filled shapes: nothing to split into parts
        panel_drawing = max(filled, key=lambda d: d["rect"].get_area())
        panel = panel_drawing["rect"]
        if part == "panel":
            return panel
        if part == "bar":
            bars = sorted(
                (d["rect"] for d in filled if d is not panel_drawing and panel.contains(d["rect"])),
                key=lambda r: r.x0,
            )
            return bars[bar - 1] if isinstance(bar, int) and 1 <= bar <= len(bars) else None

        words = [pymupdf.Rect(w[:4]) for w in self._page.get_text("words", clip=area)]
        side = {
            "title": lambda w: w.y1 <= panel.y0 and panel.y0 - w.y0 <= LABEL_REACH,
            "x_axis": lambda w: w.y0 >= panel.y1 and w.y0 - panel.y1 <= LABEL_REACH,
            "y_axis": lambda w: w.x1 <= panel.x0 and panel.x0 - w.x0 <= LABEL_REACH,
            "legend": lambda w: w.x0 >= panel.x1 and w.x0 - panel.x1 <= LEGEND_REACH,
        }.get(part)
        return _union(w for w in words if side(w)) if side else None

    def _find_quote(self, quote: str, clip: Optional[pymupdf.Rect], textpage=None) -> Optional[pymupdf.Rect]:
        """The area of ``quote`` (or of its first four words), joining matches on consecutive lines."""
        words = quote.split()
        for attempt in dict.fromkeys((" ".join(words), " ".join(words[:4]))):
            hits = self._page.search_for(attempt, clip=clip, textpage=textpage)
            if hits:
                rect = pymupdf.Rect(hits[0])
                for hit in hits[1:]:
                    if hit.y0 - rect.y1 < LINE_GAP:
                        rect |= hit
                return rect
        return None

    def _find_similar(self, quote: str, clip: Optional[pymupdf.Rect], textpage) -> Optional[pymupdf.Rect]:
        """The run of words in ``textpage`` most like ``quote``, if at least OCR_MIN_SIMILARITY alike."""
        target = " ".join(quote.lower().split())
        words = self._page.get_text("words", clip=clip, textpage=textpage, sort=True)
        best, best_ratio = None, OCR_MIN_SIMILARITY
        size = len(target.split())
        for n in {max(size - 1, 1), size, size + 1}:  # OCR can split or merge words
            for i in range(len(words) - n + 1):
                window = words[i : i + n]
                ratio = difflib.SequenceMatcher(None, target, " ".join(w[4].lower() for w in window)).ratio()
                if ratio >= best_ratio:
                    best, best_ratio = window, ratio
        return _union(w[:4] for w in best) if best else None

    def _ocr(self):
        """An OCR text page for this page, made on first use; None if Tesseract is unavailable."""
        if self._ocr_textpage is None and not PageLocator._ocr_unavailable:
            try:
                self._ocr_textpage = self._page.get_textpage_ocr(dpi=OCR_DPI, full=True, language="eng")
            except Exception as e:  # Tesseract missing or misconfigured: skip the OCR layer from now on
                PageLocator._ocr_unavailable = True
                print(f"Warning: OCR unavailable, skipping it ({e})")
        return self._ocr_textpage

    def _model_box(self, annotation: dict) -> Optional[pymupdf.Rect]:
        """The model's box in page points, if it gave a usable one on MarkUs's scale."""
        box = [annotation.get(key) for key in ("x1", "y1", "x2", "y2")]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box):
            return None
        width, height = self._page.rect.width, self._page.rect.height
        x1, y1, x2, y2 = box
        rect = pymupdf.Rect(
            min(x1, x2) / MARKUS_COORDINATE_SCALE * width,
            min(y1, y2) / MARKUS_COORDINATE_SCALE * height,
            max(x1, x2) / MARKUS_COORDINATE_SCALE * width,
            max(y1, y2) / MARKUS_COORDINATE_SCALE * height,
        ) & self._page.rect
        return rect if rect.get_area() > 0 else None

    def _to_markus(self, rect: pymupdf.Rect) -> dict:
        """``rect`` in page points as MarkUs coordinates, clamped to the page."""
        width, height = self._page.rect.width, self._page.rect.height

        def scale(value: float, size: float) -> int:
            return min(max(round(value / size * MARKUS_COORDINATE_SCALE), 0), MARKUS_COORDINATE_SCALE)

        return {
            "x1": scale(rect.x0, width),
            "y1": scale(rect.y0, height),
            "x2": scale(rect.x1, width),
            "y2": scale(rect.y1, height),
        }


def place_annotations(pdf_path: Path, page: int, reply: str) -> Optional[tuple[list[dict], list[str]]]:
    """Turn a model reply for one page into MarkUs PDF annotations.

    Returns:
        The annotations (``filename``, ``content``, ``page``, ``x1, y1, x2, y2``) and, in the
        same order, the layer that placed each one; None if the reply is not the expected JSON.
    """
    try:
        data = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        return None
    items = data.get("annotations") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return None

    annotations, placements = [], []
    with PageLocator(pdf_path, page) as locator:
        for item in items:
            if not isinstance(item, dict) or not item.get("content"):
                continue
            box, placement = locator.place(item)
            annotations.append({"filename": Path(pdf_path).name, "content": item["content"], "page": page, **box})
            placements.append(placement)
    return annotations, placements


def _is_figure_sized(rect: pymupdf.Rect) -> bool:
    return rect.width > MIN_FIGURE_SIZE and rect.height > MIN_FIGURE_SIZE


def _union(rects) -> Optional[pymupdf.Rect]:
    rects = [pymupdf.Rect(r) for r in rects]
    return functools.reduce(lambda a, b: a | b, rects) if rects else None
