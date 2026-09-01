import argparse
import json
from pathlib import Path

import fetchpaper
import parsejson
from synthesistools import PaperSynthesis, graph, save_synthesis_markdown


def main():
    # ============================================================
    # Command-line arguments
    # ============================================================
    parser = argparse.ArgumentParser(
        description="Download, parse, and synthesize an arXiv paper."
    )

    parser.add_argument(
        "paper_id",
        help="arXiv paper ID (e.g., 3101.12345)"
    )

    parser.add_argument(
        "--output",
        "-o",
        default=".",
        help="Root output directory (default: current directory)"
    )

    args = parser.parse_args()

    paper_id = args.paper_id
    root_dir = Path(args.output)

    # ============================================================
    # Define paths
    # ============================================================
    folder_dir = root_dir / paper_id
    folder_dir.mkdir(parents=True, exist_ok=True)

    tex_dir = folder_dir / f"{paper_id}.tex"
    json_path = folder_dir / f"{paper_id}.json"
    markdown_path = folder_dir / f"{paper_id}_synthesis.md"

    # ============================================================
    # Fetch Paper
    # ============================================================
    print(f"Downloading paper {paper_id}...")

    fetchpaper.download_paper(
        paper_id,
        tex_dir=str(folder_dir),
        pdf_dir=str(folder_dir)
    )

    print("Paper acquired successfully.")

    # ============================================================
    # Parse JSON
    # ============================================================
    print("Parsing LaTeX...")

    with open(tex_dir, "r", encoding="utf-8") as f:
        latex = f.read()

    parsed = parsejson.parse_latex_to_json(latex, paper_id)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    print(f"Parsed {paper_id}.json successfully.")

    # ============================================================
    # Synthesis
    # ============================================================
    print("Synthesizing paper...")

    paper = PaperSynthesis.from_json(str(json_path))

    result = graph.invoke({
        "paper": paper
    })

    paper = result["paper"]

    save_synthesis_markdown(
        paper,
        str(markdown_path)
    )

    print(f"Synthesis saved to: {markdown_path}")

    # ============================================================
    # Cleanup intermediate files
    # ============================================================
    tex_dir.unlink()
    json_path.unlink()


if __name__ == "__main__":
    main()