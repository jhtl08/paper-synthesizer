"""arXiv metadata via the public Atom API.

The Atom API is the authoritative source for a paper's title, authors,
abstract, and version. LaTeX parsing is only a fallback, because submissions
routinely use macro-heavy titles, ``authblk``, or no abstract environment at
all.
"""

import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

API_URL = "https://export.arxiv.org/api/query"

USER_AGENT = "paper-synthesizer/0.1 (+https://github.com/jhtl08/paper-synthesizer)"

_ATOM = "{http://www.w3.org/2005/Atom}"


class MetadataError(RuntimeError):
    """Raised when arXiv metadata cannot be retrieved or parsed."""


@dataclass
class ArxivMetadata:
    paper_id: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    published: str = ""
    updated: str = ""
    version: str = ""
    abs_url: str = ""
    pdf_url: str = ""

    def as_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": list(self.authors),
            "abstract": self.abstract,
            "published": self.published,
            "updated": self.updated,
            "version": self.version,
            "abs_url": self.abs_url,
            "pdf_url": self.pdf_url,
        }


def _text(node) -> str:
    """Whitespace-normalized text of an XML node."""
    if node is None or node.text is None:
        return ""
    return re.sub(r"\s+", " ", node.text).strip()


def _version_of(entry_id: str) -> str:
    match = re.search(r"v(\d+)$", entry_id)
    return match.group(0) if match else ""


def fetch_metadata(paper_id: str, timeout: int = 30) -> ArxivMetadata:
    """Look up a paper on the arXiv Atom API.

    Raises ``MetadataError`` for unknown IDs, HTTP failures, or malformed
    responses. Callers may treat that as non-fatal and fall back to metadata
    parsed out of the LaTeX source.
    """
    url = f"{API_URL}?id_list={paper_id}&max_results=1"
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/atom+xml"}
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise MetadataError(f"arXiv API returned HTTP {exc.code}: {exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise MetadataError(f"arXiv API unreachable: {exc}") from exc

    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise MetadataError(f"arXiv API returned malformed XML: {exc}") from exc

    entry = root.find(f"{_ATOM}entry")
    if entry is None:
        raise MetadataError(f"arXiv API returned no entry for {paper_id}")

    title = _text(entry.find(f"{_ATOM}title"))
    entry_id = _text(entry.find(f"{_ATOM}id"))

    # Unknown IDs come back as a synthesized error entry.
    if not title or title.lower() == "error" or "/api/errors" in entry_id:
        raise MetadataError(f"arXiv has no paper with id {paper_id}")

    authors = [
        _text(node.find(f"{_ATOM}name"))
        for node in entry.findall(f"{_ATOM}author")
    ]

    abs_url = ""
    pdf_url = ""
    for link in entry.findall(f"{_ATOM}link"):
        if link.get("rel") == "alternate" and link.get("type") == "text/html":
            abs_url = link.get("href", "")
        elif link.get("title") == "pdf":
            pdf_url = link.get("href", "")

    return ArxivMetadata(
        paper_id=paper_id,
        title=title,
        authors=[author for author in authors if author],
        abstract=_text(entry.find(f"{_ATOM}summary")),
        published=_text(entry.find(f"{_ATOM}published")),
        updated=_text(entry.find(f"{_ATOM}updated")),
        version=_version_of(entry_id),
        abs_url=abs_url or f"https://arxiv.org/abs/{paper_id}",
        pdf_url=pdf_url or f"https://arxiv.org/pdf/{paper_id}.pdf",
    )
