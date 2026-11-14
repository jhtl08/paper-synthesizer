import io
import tarfile
import urllib.error

import pytest

import fetchpaper


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(url, code, headers=None):
    return urllib.error.HTTPError(url, code, "error", headers or {}, None)


def make_tar(members):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_fetch_retries_on_429(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        if len(calls) < 3:
            raise http_error(request.full_url, 429, {"Retry-After": "0"})
        return FakeResponse(b"payload")

    monkeypatch.setattr(fetchpaper.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(fetchpaper.time, "sleep", lambda seconds: None)

    assert fetchpaper._fetch("https://example.test/x", attempts=3) == b"payload"
    assert len(calls) == 3


def test_fetch_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(
        fetchpaper.urllib.request,
        "urlopen",
        lambda request, timeout=None: (_ for _ in ()).throw(
            http_error(request.full_url, 503)
        ),
    )
    monkeypatch.setattr(fetchpaper.time, "sleep", lambda seconds: None)

    with pytest.raises(fetchpaper.FetchError) as excinfo:
        fetchpaper._fetch("https://example.test/x", attempts=3)

    assert excinfo.value.status == 503


def test_fetch_does_not_retry_404(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        raise http_error(request.full_url, 404)

    monkeypatch.setattr(fetchpaper.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(fetchpaper.FetchError) as excinfo:
        fetchpaper._fetch("https://example.test/x", attempts=3)

    assert excinfo.value.status == 404
    assert len(calls) == 1


def test_tex_download_keeps_sibling_pdf(tmp_path, monkeypatch):
    # Regression: the old extractor deleted every other file in the directory.
    (tmp_path / "1234.5678.pdf").write_bytes(b"existing pdf")
    monkeypatch.setattr(
        fetchpaper,
        "_fetch",
        lambda url, **kwargs: rb"\documentclass{article}\begin{document}ok\end{document}",
    )

    fetchpaper.download_tex("1234.5678", output_dir=str(tmp_path))

    assert (tmp_path / "1234.5678.pdf").read_bytes() == b"existing pdf"
    assert (tmp_path / "1234.5678.tex").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".psynth-")]


def test_malicious_tar_is_contained(tmp_path):
    payload = make_tar(
        [
            ("../escaped.txt", b"pwned"),
            ("main.tex", b"\\documentclass{article}"),
        ]
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with pytest.raises(fetchpaper.NoSourceError):
        fetchpaper._read_source_payload(payload, scratch)

    assert not (tmp_path / "escaped.txt").exists()


def test_download_paper_falls_back_to_pdf(tmp_path, monkeypatch):
    class FakeMetadata:
        def as_dict(self):
            return {"title": "T", "authors": ["A"], "abstract": "Abs"}

    monkeypatch.setattr(fetchpaper.arxivmeta, "fetch_metadata", lambda *a, **k: FakeMetadata())
    monkeypatch.setattr(
        fetchpaper, "download_tex", lambda *a, **k: (_ for _ in ()).throw(
            fetchpaper.NoSourceError("no tex")
        )
    )

    def fake_pdf(paper_id, output_dir="."):
        path = tmp_path / f"{paper_id}.pdf"
        path.write_bytes(b"%PDF-1.4")
        return path

    monkeypatch.setattr(fetchpaper, "download_pdf", fake_pdf)

    result = fetchpaper.download_paper("9999.0001", tex_dir=str(tmp_path), pdf_dir=str(tmp_path))

    assert result.kind == "pdf"
    assert result.pdf_path.exists()
    assert result.metadata["title"] == "T"
