"""arXiv downloads: PDF, LaTeX source archives, and orchestration."""

import gzip
import logging
import os
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import arxivmeta
import texsource

log = logging.getLogger(__name__)

BASE_URL = "https://arxiv.org"

# Codes worth retrying: transient server-side failures and throttling.
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3
TIMEOUT = 30


class FetchError(RuntimeError):
    """A download failed."""

    def __init__(self, message, status=None, url=None):
        super().__init__(message)
        self.status = status
        self.url = url


class NoSourceError(FetchError):
    """arXiv has no TeX source for this paper (PDF-only submission)."""


@dataclass
class FetchResult:
    """What we managed to obtain for a paper."""

    paper_id: str
    kind: str = "none"  # "tex", "pdf", or "none"
    tex_path: Path | None = None
    pdf_path: Path | None = None
    metadata: dict = field(default_factory=dict)


def _retry_delay(exc: urllib.error.HTTPError, attempt: int) -> int:
    """Seconds to wait, honouring Retry-After when arXiv sends one."""
    header = exc.headers.get("Retry-After") if exc.headers else None
    if header:
        try:
            return max(0, min(int(header), 60))
        except ValueError:
            pass
    return min(2 ** (attempt - 1), 30)


def _fetch(url: str, *, timeout: int = TIMEOUT, attempts: int = MAX_ATTEMPTS) -> bytes:
    """Download a URL with a User-Agent, timeout, and bounded retries.

    Raises ``FetchError`` for non-retryable HTTP errors and for exhausted
    retries. Error pages are never returned as if they were payloads.
    """
    last_error = None

    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": arxivmeta.USER_AGENT, "Accept": "*/*"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in RETRY_STATUS and attempt < attempts:
                delay = _retry_delay(exc, attempt)
                log.debug("HTTP %s from %s, retrying in %ss", exc.code, url, delay)
                time.sleep(delay)
                continue
            raise FetchError(
                f"HTTP {exc.code} for {url}: {exc.reason}",
                status=exc.code,
                url=url,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < attempts:
                delay = min(2 ** (attempt - 1), 30)
                log.debug("network error from %s (%s), retrying in %ss", url, exc, delay)
                time.sleep(delay)
                continue
            raise FetchError(f"network error for {url}: {exc}", url=url) from exc

    raise FetchError(f"failed to download {url}: {last_error}", url=url)


def _write_atomic(path: Path, data: bytes) -> Path:
    """Write bytes via a .part file so a failed run never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def download_pdf(paper_id: str, output_dir: str = ".") -> Path:
    """Download the PDF for ``paper_id`` atomically."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    payload = _fetch(f"{BASE_URL}/pdf/{paper_id}.pdf")
    final_path = _write_atomic(output / f"{paper_id}.pdf", payload)

    log.info("Downloaded PDF: %s", final_path)
    return final_path


def _read_source_payload(payload: bytes, scratch: Path) -> str:
    """Turn an e-print payload (archive, gzipped TeX, or plain TeX) into text."""
    scratch.mkdir(parents=True, exist_ok=True)
    payload_path = scratch / "source.bin"
    payload_path.write_bytes(payload)

    try:
        if tarfile.is_tarfile(payload_path):
            with tarfile.open(payload_path, "r:*") as archive:
                # "data" rejects absolute paths, parent traversal, device
                # files, and other members that could escape the scratch dir.
                archive.extractall(scratch, filter="data")

            main_tex = texsource.find_main_tex(scratch)
            if main_tex is None:
                raise NoSourceError("arXiv archive contains no .tex file")

            latex = main_tex.read_text(encoding="utf-8", errors="replace")
            # Papers split prose into \input{...} files: resolve them so the
            # parser sees the whole document, not just the preamble stub.
            return texsource.inline_inputs(latex, main_tex.parent)

        if payload[:2] == b"\x1f\x8b":
            return gzip.decompress(payload).decode("utf-8", errors="replace")

        if b"\\documentclass" in payload or b"\\begin{document}" in payload:
            return payload.decode("utf-8", errors="replace")

        raise NoSourceError("arXiv returned neither a TeX archive nor a TeX file")

    except tarfile.TarError as exc:
        raise NoSourceError(f"unreadable TeX archive: {exc}") from exc


def download_tex(paper_id: str, output_dir: str = ".") -> Path:
    """Download a paper's LaTeX source and leave exactly one ``.tex`` file.

    Extraction happens in a scratch directory, so nothing already present in
    ``output_dir`` is touched or removed.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / f"{paper_id}.tex"

    payload = _fetch(f"{BASE_URL}/e-print/{paper_id}")

    scratch = Path(tempfile.mkdtemp(prefix=f".psynth-{paper_id}-", dir=output))
    try:
        latex = _read_source_payload(payload, scratch)
        final_path.write_text(latex, encoding="utf-8")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    log.info("Downloaded TeX: %s", final_path)
    return final_path


def download_paper(paper_id: str, tex_dir: str = None, pdf_dir: str = None) -> FetchResult:
    """Fetch metadata, TeX source, and PDF, degrading to PDF-only when needed.

    The returned ``FetchResult`` tells the caller which representation is
    available instead of forcing it to infer that from the filesystem.
    """
    tex_root = Path(tex_dir or paper_id)
    pdf_root = Path(pdf_dir or paper_id)
    tex_root.mkdir(parents=True, exist_ok=True)
    pdf_root.mkdir(parents=True, exist_ok=True)

    result = FetchResult(paper_id=paper_id)

    try:
        result.metadata = arxivmeta.fetch_metadata(paper_id).as_dict()
    except arxivmeta.MetadataError as exc:
        log.warning("metadata lookup failed (%s); will fall back to LaTeX", exc)

    tex_path = tex_root / f"{paper_id}.tex"
    pdf_path = pdf_root / f"{paper_id}.pdf"

    if tex_path.exists():
        log.info("TeX already exists, skipping download")
        result.tex_path = tex_path
        result.kind = "tex"
    else:
        try:
            result.tex_path = download_tex(paper_id, output_dir=str(tex_root))
            result.kind = "tex"
        except NoSourceError as exc:
            log.warning("no TeX source (%s)", exc)
        except FetchError as exc:
            if exc.status in (400, 404):
                log.warning("no TeX source (HTTP %s); arXiv may only host a PDF", exc.status)
            else:
                raise

    if pdf_path.exists():
        result.pdf_path = pdf_path
    else:
        result.pdf_path = download_pdf(paper_id, output_dir=str(pdf_root))

    if result.kind != "tex":
        result.kind = "pdf" if result.pdf_path else "none"

    return result
