import synthesistools as s


def build_paper():
    paper = s.PaperSynthesis(
        paper_id="2501.00001",
        title="A Study of Things",
        authors=["Ada Lovelace", "Alan Turing"],
        abstract="We study things.",
        conclusion="Things are studied.",
        source="tex",
        arxiv_url="https://arxiv.org/abs/2501.00001v2",
        version="v2",
        model="deepseek-v4.1-flash",
        generated_at="2026-01-02T03:04:05+00:00",
    )
    paper.overarching_synthesis = "Overview."
    paper.knowledge_gap = "Gap."
    paper.hypothesis = "Hypothesis."
    paper.results = "Results."
    paper.implications = "Implications."
    paper.running_synthesis = "Running."

    for index in range(3):
        section = s.Section(heading=f"Section {index}", content=f"Body {index}")
        paper.sections.append(section)

    paper.sections[0].synthesis = "First synthesis."
    paper.sections[0].remarks = "First remarks."

    return paper


def read_frontmatter(text):
    assert text.startswith("---\n")
    _, block, _ = text.split("---", 2)
    return {
        line.split(":", 1)[0]: line.split(":", 1)[1].strip()
        for line in block.strip().splitlines()
        if ":" in line and not line.startswith("  ")
    }


def test_markdown_has_frontmatter_and_body(tmp_path, capsys):
    paper = build_paper()
    out = tmp_path / "out.md"

    returned = s.save_synthesis_markdown(paper, str(out))

    assert returned == str(out)
    assert capsys.readouterr().out == ""  # caller owns the confirmation

    text = out.read_text(encoding="utf-8")
    meta = read_frontmatter(text)

    assert meta["paper_id"] == '"2501.00001"'
    assert meta["source"] == '"tex"'
    assert meta["model"] == '"deepseek-v4.1-flash"'
    assert meta["generated_at"] == '"2026-01-02T03:04:05+00:00"'
    assert meta["sections_synthesized"] == "1"
    assert meta["sections_available"] == "3"
    assert '- "Ada Lovelace"' in text

    for heading in [
        "## Abstract",
        "## Overarching Synthesis",
        "## Knowledge Gap",
        "## Hypothesis",
        "## Results",
        "## Implications",
        "## Final Running Synthesis",
        "## Individual Section Synthesis and Remarks",
    ]:
        assert heading in text


def test_skipped_sections_are_listed(tmp_path):
    paper = build_paper()
    out = tmp_path / "out.md"

    s.save_synthesis_markdown(paper, str(out))
    text = out.read_text(encoding="utf-8")

    assert "## Not Synthesized" in text
    assert "- Section 1" in text
    assert "- Section 2" in text

    # Synthesized sections are rendered, skipped ones are not.
    assert "### [0] Section 0" in text
    assert "### [1] Section 1" not in text


def test_no_skipped_section_when_all_synthesized(tmp_path):
    paper = build_paper()
    for section in paper.sections:
        section.synthesis = "Synth."
        section.remarks = "Remark."

    out = tmp_path / "out.md"
    s.save_synthesis_markdown(paper, str(out))

    assert "## Not Synthesized" not in out.read_text(encoding="utf-8")


def test_authors_frontmatter_is_yaml_list(tmp_path):
    paper = build_paper()
    out = tmp_path / "out.md"
    s.save_synthesis_markdown(paper, str(out))

    text = out.read_text(encoding="utf-8")
    block = text.split("---")[1]
    assert "authors:\n  - \"Ada Lovelace\"\n  - \"Alan Turing\"" in block
