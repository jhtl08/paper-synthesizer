import logging

import texsource


def build_tree(root):
    (root / "sec").mkdir(parents=True)
    (root / "main.tex").write_text(
        r"""\documentclass{article}
\begin{document}
\input{sec/intro}
\input{sec/missing}
\end{document}""",
        encoding="utf-8",
    )
    (root / "sec" / "intro.tex").write_text(
        r"\section{Introduction}\input{deep}", encoding="utf-8"
    )
    (root / "sec" / "deep.tex").write_text(
        r"DEEP TEXT \input{intro}", encoding="utf-8"
    )
    return root


def test_find_main_tex_prefers_documentclass(tmp_path):
    build_tree(tmp_path)
    (tmp_path / "sec" / "huge.tex").write_text("x" * 5000, encoding="utf-8")

    assert texsource.find_main_tex(tmp_path).name == "main.tex"


def test_find_main_tex_returns_none_without_tex(tmp_path):
    (tmp_path / "readme.txt").write_text("no tex here", encoding="utf-8")
    assert texsource.find_main_tex(tmp_path) is None


def test_inline_inputs_resolves_nested_includes(tmp_path, caplog):
    build_tree(tmp_path)
    main = texsource.find_main_tex(tmp_path)

    with caplog.at_level(logging.WARNING):
        out = texsource.inline_inputs(main.read_text(encoding="utf-8"), main.parent)

    assert "\\section{Introduction}" in out
    assert "DEEP TEXT" in out
    assert "sec/missing" in out  # unresolved include is preserved
    assert "cyclic include skipped" in out
    assert any("unresolved" in record.message for record in caplog.records)


def test_inline_inputs_refuses_paths_outside_tree(tmp_path, caplog):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "main.tex").write_text(r"\input{../secret}", encoding="utf-8")
    (tmp_path / "secret.tex").write_text("SECRET", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        out = texsource.inline_inputs((root / "main.tex").read_text(), root)

    assert "SECRET" not in out
    assert any("outside source tree" in record.message for record in caplog.records)


def test_split_pdf_text_uses_numbered_headings():
    text = """1. Introduction
Some intro prose.

2. Method
We did a thing.

2.1. Setup
Details here.

3. Results
It worked.

References
[1] Someone.
"""
    sections = texsource.split_pdf_text(text)

    assert [section["heading"] for section in sections][:3] == [
        "1. Introduction",
        "2. Method",
        "2.1. Setup",
    ]
    assert "References" not in [section["heading"] for section in sections]


def test_split_pdf_text_falls_back_to_single_section():
    text = "Just a wall of text with no recognizable headings at all."
    sections = texsource.split_pdf_text(text)

    assert len(sections) == 1
    assert sections[0]["heading"] == "Full Text"
    assert sections[0]["content"] == text
