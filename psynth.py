import argparse
import json
import logging
from pathlib import Path

import fetchpaper
import parsejson
import texsource
from synthesistools import (
    PaperSynthesis,
    graph,
    preflight,
    save_synthesis_markdown,
    set_max_section_chars,
)

log = logging.getLogger("psynth")


def _fail(message: str, code: int = 1) -> None:
    log.error("%s", message)
    raise SystemExit(code)


def _apply_metadata(parsed: dict, metadata: dict) -> None:
    """Prefer arXiv API metadata over values guessed from the LaTeX source."""
    if not metadata:
        return

    parsed["title"] = metadata.get("title") or parsed.get("title") or ""
    if metadata.get("authors"):
        parsed["authors"] = metadata["authors"]
    if metadata.get("abstract"):
        parsed["abstract"] = metadata["abstract"]

    parsed["arxiv_url"] = metadata.get("abs_url", "")
    parsed["version"] = metadata.get("version", "")


def _parsed_from_tex(tex_path: Path, paper_id: str) -> dict:
    latex = tex_path.read_text(encoding="utf-8", errors="replace")
    parsed = parsejson.parse_latex_to_json(latex, paper_id)
    parsed["source"] = "tex"
    return parsed


def _parsed_from_pdf(pdf_path: Path, paper_id: str, metadata: dict) -> dict:
    text = texsource.pdf_to_text(pdf_path)
    if not text.strip():
        _fail(f"could not extract any text from {pdf_path}")

    return {
        "paper_id": paper_id,
        "title": metadata.get("title", ""),
        "authors": metadata.get("authors", []),
        "abstract": metadata.get("abstract", ""),
        "section": texsource.split_pdf_text(text),
        "source": "pdf-text",
    }


def _build_parsed(paper_id: str, folder_dir: Path, result: fetchpaper.FetchResult) -> dict:
    """Parse the best representation available: TeX first, PDF text second."""
    if result.kind == "tex" and result.tex_path and result.tex_path.exists():
        parsed = _parsed_from_tex(result.tex_path, paper_id)
        _apply_metadata(parsed, result.metadata)

        if parsed["section"]:
            return parsed

        log.warning("TeX source parsed to zero sections")
        if not result.pdf_path:
            _fail(
                "No sections found in the TeX source. The paper may use an "
                "unsupported structure; open the .tex file to inspect it."
            )

        log.warning("falling back to PDF text extraction")
        parsed = _parsed_from_pdf(result.pdf_path, paper_id, result.metadata)
        _apply_metadata(parsed, result.metadata)
        return parsed

    if result.pdf_path and result.pdf_path.exists():
        log.warning("no TeX source available; using PDF text (lower fidelity)")
        parsed = _parsed_from_pdf(result.pdf_path, paper_id, result.metadata)
        _apply_metadata(parsed, result.metadata)
        return parsed

    _fail(f"could not obtain a TeX source or PDF for {paper_id}")

    return {}  # unreachable, keeps type checkers happy


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download, parse, and synthesize an arXiv paper."
    )

    parser.add_argument(
        "paper_id",
        nargs="?",
        help="arXiv paper ID (e.g., 3101.12345)"
    )

    parser.add_argument(
        "--output",
        "-o",
        default=".",
        help="Root output directory (default: current directory)"
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the API key and model access, then exit"
    )

    parser.add_argument(
        "--max-section-chars",
        type=int,
        default=24000,
        help="Truncate each section before prompting (default: 24000, 0 = no limit)"
    )

    parser.add_argument(
        "--all-sections",
        action="store_true",
        help="Synthesize every section instead of only the selected ones"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <paper_id>.state.json, skipping completed sections"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show request-level detail (DEBUG logging)"
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.check:
        try:
            info = preflight()
        except RuntimeError as exc:
            _fail(str(exc), 2)
        print(f"OK: {info['model']} is available at {info['base_url']}")
        return

    if not args.paper_id:
        parser.error("paper_id is required unless --check is used")

    paper_id = args.paper_id
    root_dir = Path(args.output)

    set_max_section_chars(args.max_section_chars)

    folder_dir = root_dir / paper_id
    folder_dir.mkdir(parents=True, exist_ok=True)

    json_path = folder_dir / f"{paper_id}.json"
    markdown_path = folder_dir / f"{paper_id}_synthesis.md"
    state_path = folder_dir / f"{paper_id}.state.json"

    # ============================================================
    # Fail fast on auth/model problems before downloading anything
    # ============================================================
    try:
        preflight()
    except RuntimeError as exc:
        _fail(str(exc), 2)

    # ============================================================
    # Fetch: metadata + TeX (or PDF) in one place
    # ============================================================
    print(f"Downloading paper {paper_id}...", flush=True)

    result = fetchpaper.download_paper(
        paper_id,
        tex_dir=str(folder_dir),
        pdf_dir=str(folder_dir),
    )

    # ============================================================
    # Parse
    # ============================================================
    print("Parsing paper...", flush=True)
    parsed = _build_parsed(paper_id, folder_dir, result)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    print(f"Parsed {len(parsed['section'])} sections from {parsed['source']}.", flush=True)

    # ============================================================
    # Synthesis
    # ============================================================
    print("Synthesizing paper...", flush=True)

    paper = PaperSynthesis.from_json(str(json_path))

    if args.resume and state_path.exists():
        try:
            saved = PaperSynthesis.model_validate_json(
                state_path.read_text(encoding="utf-8")
            )
        except ValueError as exc:
            log.warning("ignoring unreadable state file (%s)", exc)
        else:
            if [s.heading for s in saved.sections] == [s.heading for s in paper.sections]:
                done = sum(1 for section in saved.sections if section.synthesis)
                print(f"Resuming from state: {done} section(s) already synthesized.")
                paper = saved
            else:
                log.warning(
                    "state file does not match the parsed paper; starting fresh"
                )
    elif args.resume:
        log.warning("no state file found; starting fresh")

    result_state = graph.invoke({
        "paper": paper,
        "all_sections": args.all_sections,
        "state_path": str(state_path),
    })

    paper = result_state["paper"]

    saved_path = save_synthesis_markdown(
        paper,
        str(markdown_path)
    )

    print(f"Synthesis saved to: {saved_path}", flush=True)

    # ============================================================
    # Cleanup intermediate files
    # ============================================================
    for intermediate in (result.tex_path, json_path, state_path):
        if intermediate and Path(intermediate).exists():
            Path(intermediate).unlink()


if __name__ == "__main__":
    main()
