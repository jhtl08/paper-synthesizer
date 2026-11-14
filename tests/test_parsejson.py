import parsejson

SAMPLE = r"""
\documentclass{article}
\title{Learning \textbf{Deep} Models with \emph{Noise} \thanks{thanks}}
\author{Ada Lovelace \and Alan Turing}
\begin{document}
\begin{abstract}
We study a problem \citep[see][p.~3]{smith2020} and show \textbf{results}.
\end{abstract}

\section{Introduction}
Prior work \cite{doe2021} shows this. See Section~\ref{sec:2} and \eqref{eq:1}.
A footnote\footnote{Ignore me.} and \textit{emphasis} with $x^2 + y^2$.

\subsection{Setup}
Setup prose.

\subsubsection{Devices}
Device prose.

\section{Method}
Method prose.

\section{Conclusion}
We conclude.
\end{document}
"""


def test_metadata_and_hierarchy():
    parsed = parsejson.parse_latex_to_json(SAMPLE, "test.0001")

    assert parsed["paper_id"] == "test.0001"
    assert parsed["title"] == "Learning Deep Models with Noise"
    assert parsed["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert parsed["abstract"] == "We study a problem and show results."

    headings = [section["heading"] for section in parsed["section"]]
    assert headings == ["Introduction", "Method", "Conclusion"]

    intro = parsed["section"][0]
    assert intro["subsection"][0]["subheading"] == "Setup"
    assert intro["subsection"][0]["subsubsection"][0]["subsubheading"] == "Devices"
    assert intro["subsection"][0]["subsubsection"][0]["subsubcontent"] == "Device prose."


def test_markup_is_stripped():
    parsed = parsejson.parse_latex_to_json(SAMPLE, "test.0001")
    body = parsed["section"][0]["content"]

    assert "\\cite" not in body
    assert "\\ref" not in body
    assert "\\footnote" not in body
    assert "\\textbf" not in body
    assert "~" not in body
    # Inline math is content, not markup: leave it alone.
    assert "$x^2 + y^2$" in body


def test_comments_are_removed():
    latex = r"""
\documentclass{article}
\begin{document}
\section{Intro}
Visible text. % hidden comment
More visible.
\end{document}
"""
    parsed = parsejson.parse_latex_to_json(latex, "x")
    assert "hidden comment" not in parsed["section"][0]["content"]
    assert "Visible text." in parsed["section"][0]["content"]


def test_missing_abstract_is_empty():
    latex = r"\documentclass{article}\begin{document}\section{Intro}Body.\end{document}"
    parsed = parsejson.parse_latex_to_json(latex, "x")
    assert parsed["abstract"] == ""


def test_zero_sections():
    latex = r"\documentclass{article}\begin{document}Just prose, no headings.\end{document}"
    parsed = parsejson.parse_latex_to_json(latex, "x")
    assert parsed["section"] == []


def test_clean_latex_text_unwraps_nested_formatting():
    cleaned = parsejson.clean_latex_text(r"Nested \textbf{bold \emph{both}} text.")
    assert cleaned == "Nested bold both text."


def test_layout_commands_are_removed():
    cleaned = parsejson.clean_latex_text(
        r"Real text. \vspace{-5pt} \noindent More text. \newpage"
    )
    assert "\\vspace" not in cleaned
    assert "\\noindent" not in cleaned
    assert "\\newpage" not in cleaned
    assert "Real text." in cleaned and "More text." in cleaned
