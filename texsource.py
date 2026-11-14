"""LaTeX source preparation and PDF text extraction.

arXiv submissions are frequently multi-file: ``main.tex`` contains only
``\\documentclass``, front matter, and a list of ``\\input{}`` calls, with the
actual prose in sibling files. Parsing the main file alone yields zero
sections, so includes are resolved before parsing.

PDF text extraction is the fallback for papers that have no TeX source at all.
"""

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

INCLUDE_PATTERN = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
MAX_INCLUDE_DEPTH = 10


class PdfTextError(RuntimeError):
    """Raised when a PDF cannot be converted to text."""


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def find_main_tex(root: Path) -> Path | None:
    """Pick the root ``.tex`` file of an extracted source tree.

    Prefers files that declare a document class, then falls back to the
    largest file. Returns ``None`` when the tree has no TeX at all.
    """
    root = Path(root)
    candidates = [path for path in root.rglob("*.tex") if path.is_file()]
    if not candidates:
        return None

    roots = [path for path in candidates if "\\documentclass" in _read(path)]
    pool = roots or candidates
    return max(pool, key=lambda path: path.stat().st_size)


def _resolve_include(root: Path, name: str, base: Path | None = None) -> Path | None:
    """Resolve an ``\\input`` argument against the root and the including file."""
    name = name.strip()
    if not name:
        return None

    # LaTeX resolves includes against the main file's directory, but authors
    # also use paths relative to the file containing the \input. Try both.
    directories = [root]
    if base is not None and base.resolve() != root:
        directories.append(base.resolve())

    for directory in directories:
        target = (directory / name).resolve()
        candidates = [target] if target.suffix == ".tex" else [target.with_suffix(".tex"), target]

        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


def inline_inputs(
    text: str,
    root: Path,
    _seen: set | None = None,
    _depth: int = 0,
    _base: Path | None = None,
) -> str:
    """Recursively replace ``\\input{}`` / ``\\include{}`` with file contents.

    Guards against cycles, runaway depth, and paths that resolve outside the
    extracted source tree. Unresolved includes are left in place and logged,
    since a missing file should not discard the rest of the paper.
    """
    root = Path(root).resolve()
    base = root if _base is None else Path(_base).resolve()
    seen = set() if _seen is None else _seen

    if _depth > MAX_INCLUDE_DEPTH:
        log.warning("include depth limit (%s) reached; leaving includes as-is", MAX_INCLUDE_DEPTH)
        return text

    def replace(match: re.Match) -> str:
        name = match.group(1)
        path = _resolve_include(root, name, base)

        if path is None:
            log.warning("unresolved \\input{%s}", name)
            return match.group(0)

        try:
            path.relative_to(root)
        except ValueError:
            log.warning("refusing include outside source tree: %s", name)
            return match.group(0)

        if path in seen:
            log.warning("skipping cyclic include: %s", name)
            return f"% [psynth] cyclic include skipped: {name}"

        seen.add(path)
        body = _read(path)
        if not body:
            log.warning("include is empty or unreadable: %s", name)
            return ""

        return inline_inputs(body, root, seen, _depth + 1, path.parent)

    return INCLUDE_PATTERN.sub(replace, text)


def pdf_to_text(path: Path, max_pages: int = 80) -> str:
    """Extract text from a PDF, capped so a 300-page thesis stays tractable."""
    try:
        from pdfminer.high_level import extract_text
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise PdfTextError(
            "pdfminer.six is required for PDF-only papers; install requirements"
        ) from exc

    try:
        text = extract_text(str(path), page_numbers=list(range(max_pages)))
    except Exception as exc:  # pdfminer raises a wide range of parser errors
        raise PdfTextError(f"could not extract text from {path}: {exc}") from exc

    text = text.replace("\f", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# A numbered heading ("3 Method", "3.1 Setup") or a well-known unnumbered one.
_NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[IVX]+\.)\s+\S")
_KNOWN_HEADINGS = {
    "abstract", "introduction", "background", "related work", "method",
    "methods", "methodology", "approach", "experiments", "experimental setup",
    "results", "evaluation", "analysis", "discussion", "limitations",
    "conclusion", "conclusions", "future work",
}
_STOP_HEADINGS = {"references", "bibliography", "acknowledgments", "acknowledgements"}


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not 3 <= len(stripped) <= 90:
        return False
    if stripped.endswith((".", ",", ";", "@")) or "http" in stripped:
        return False
    if len(stripped.split()) > 12:
        return False
    lowered = stripped.lower().rstrip(":")
    return bool(_NUMBERED_HEADING.match(stripped)) or lowered in _KNOWN_HEADINGS


def split_pdf_text(text: str, max_sections: int = 12) -> list[dict]:
    """Approximate a section hierarchy from extracted PDF text.

    PDFs carry no structure, so headings are detected heuristically. When the
    heuristic finds fewer than two sections, the whole text becomes a single
    ``Full Text`` section — the pipeline still works, just coarser.
    """
    sections: list[tuple[str, list[str]]] = []
    heading: str | None = None
    buffer: list[str] = []

    for line in text.splitlines():
        if _looks_like_heading(line):
            lowered = line.strip().lower().rstrip(":")
            if lowered in _STOP_HEADINGS and sections:
                break

            if heading is not None:
                sections.append((heading, buffer))
            heading = line.strip()
            buffer = []
        elif heading is not None:
            buffer.append(line)

    if heading is not None:
        sections.append((heading, buffer))

    parsed = [
        {
            "heading": name,
            "content": re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip(),
            "subsection": [],
        }
        for name, lines in sections[:max_sections]
    ]
    parsed = [section for section in parsed if section["content"]]

    if len(parsed) < 2:
        return [{"heading": "Full Text", "content": text.strip(), "subsection": []}]

    return parsed
