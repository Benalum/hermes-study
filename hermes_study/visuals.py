from __future__ import annotations

from pathlib import Path


class VisualUnavailable(RuntimeError):
    pass


def render_pdf_page(
    source_path: Path,
    page_number: int,
    cache_dir: Path,
    *,
    cache_key: str,
    scale: float = 1.6,
) -> Path:
    """Render a 1-based PDF page to a cached PNG for browser display."""
    source_path = Path(source_path)
    if source_path.suffix.lower() != ".pdf":
        raise VisualUnavailable("Referenced source is not a PDF.")
    if not source_path.exists():
        raise VisualUnavailable("Referenced PDF is no longer available on disk.")
    if page_number < 1:
        raise VisualUnavailable("Page numbers are 1-based and must be positive.")

    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        raise VisualUnavailable(
            "PDF visual rendering is not installed. Re-run scripts/install_mac.sh."
        ) from exc

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_key = "".join(ch for ch in cache_key if ch.isalnum() or ch in "-_" )[:80] or "document"
    output = cache_dir / f"{safe_key}-p{page_number}.png"
    if output.exists() and output.stat().st_mtime >= source_path.stat().st_mtime:
        return output

    try:
        with fitz.open(source_path) as pdf:
            if page_number > pdf.page_count:
                raise VisualUnavailable(
                    f"Page {page_number} is outside this PDF, which has {pdf.page_count} pages."
                )
            page = pdf.load_page(page_number - 1)
            matrix = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(output)
    except VisualUnavailable:
        raise
    except Exception as exc:
        raise VisualUnavailable(f"Could not render PDF page {page_number}: {exc}") from exc

    return output
