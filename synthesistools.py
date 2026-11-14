import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# ============================================================
# Subsubsection
# ============================================================

class Subsubsection(BaseModel):
    subsubheading: str
    subsubcontent: str
    synthesis: str = ""
    remarks: str = ""


# ============================================================
# Subsection
# ============================================================

class Subsection(BaseModel):
    subheading: str
    subcontent: str
    synthesis: str = ""
    remarks: str = ""
    subsubsection: list[Subsubsection] = Field(default_factory=list)


# ============================================================
# Section
# ============================================================

class Section(BaseModel):
    heading: str
    content: str
    synthesis: str = ""
    remarks: str = ""
    running_synthesis: str = ""
    subsection: list[Subsection] = Field(default_factory=list)


# ============================================================
# Paper Synthesis
# ============================================================

class PaperSynthesis(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    conclusion: str

    sections: list[Section] = Field(default_factory=list)

    # Provenance, filled from the arXiv API / CLI
    source: str = "tex"
    arxiv_url: str = ""
    version: str = ""
    model: str = ""
    generated_at: str = ""

    # Pass 1
    overarching_synthesis: str = ""

    # Pass 1.5
    relevant_section_indices: list[int] = Field(default_factory=list)

    # Pass 2
    hypothesis: str = ""
    knowledge_gap: str = ""
    results: str = ""
    implications: str = ""

    # Pass 3
    running_synthesis: str = ""

    @classmethod
    def from_json(cls, json_path: str):
        """Initialize PaperSynthesis from a parsed paper JSON file."""

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # ----------------------------------------------------
        # Find and compile the conclusion
        # ----------------------------------------------------

        conclusion = ""

        for section in data.get("section", []):

            # Match headings beginning with "Conclusion"
            if section.get("heading", "").strip().lower().startswith("conclusion"):

                conclusion_parts = []

                # Section content
                if section.get("content"):
                    conclusion_parts.append(section["content"])

                # Subsections and subsubsections
                for subsection in section.get("subsection", []):

                    if subsection.get("subheading"):
                        conclusion_parts.append(
                            subsection["subheading"]
                        )

                    if subsection.get("subcontent"):
                        conclusion_parts.append(
                            subsection["subcontent"]
                        )

                    for subsubsection in subsection.get(
                        "subsubsection", []
                    ):

                        if subsubsection.get("subsubheading"):
                            conclusion_parts.append(
                                subsubsection["subsubheading"]
                            )

                        if subsubsection.get("subsubcontent"):
                            conclusion_parts.append(
                                subsubsection["subsubcontent"]
                            )

                conclusion = "\n\n".join(conclusion_parts)
                break

        # ----------------------------------------------------
        # No conclusion found
        # ----------------------------------------------------

        if not conclusion:
            conclusion = (
                "Infer Conclusion from the abstract and later sections "
                "of the paper."
            )

        # ----------------------------------------------------
        # Create PaperSynthesis object
        # ----------------------------------------------------

        return cls(
            paper_id=data.get("paper_id", ""),
            title=data.get("title", ""),
            authors=data.get("authors", []),
            abstract=data.get("abstract", ""),
            conclusion=conclusion,
            sections=data.get("section", []),
            source=data.get("source", "tex"),
            arxiv_url=data.get("arxiv_url", ""),
            version=data.get("version", ""),
            model=data.get("model") or MODEL,
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

# ============================================================
# Helper Functions
# ============================================================

def get_paper_structure(paper: PaperSynthesis) -> str:
    structure = []

    for section in paper.sections:
        structure.append(section.heading)

        for subsection in section.subsection:
            structure.append(f"  {subsection.subheading}")

            for subsubsection in subsection.subsubsection:
                structure.append(f"    {subsubsection.subsubheading}")

    return "\n".join(structure)

def get_indexed_paper_structure(paper: PaperSynthesis) -> str:
    structure = []

    for i, section in enumerate(paper.sections):
        structure.append(f"[{i}] {section.heading}")

        for subsection in section.subsection:
            structure.append(f"    - {subsection.subheading}")

            for subsubsection in subsection.subsubsection:
                structure.append(
                    f"        - {subsubsection.subsubheading}"
                )

    return "\n".join(structure)

def select_sections(paper: PaperSynthesis, all_sections: bool = False) -> list[Section]:
    """Pick the sections Pass 3 should synthesize.

    By default this honours the Pass 1.5 selection; ``--all-sections`` (or an
    empty/unusable selection) falls back to the full paper.
    """
    if all_sections or not paper.relevant_section_indices:
        return list(paper.sections)

    selected = [
        paper.sections[index]
        for index in paper.relevant_section_indices
        if 0 <= index < len(paper.sections)
    ]

    return selected or list(paper.sections)


def get_relevant_section_content(paper: PaperSynthesis) -> str:
    content = []

    for i in paper.relevant_section_indices:
        if 0 <= i < len(paper.sections):
            section = paper.sections[i]

            section_text = f"SECTION: {section.heading}\n"

            if section.content:
                section_text += section.content + "\n"

            for subsection in section.subsection:
                section_text += f"\nSUBSECTION: {subsection.subheading}\n"

                if subsection.subcontent:
                    section_text += subsection.subcontent + "\n"

                for subsubsection in subsection.subsubsection:
                    section_text += (
                        f"\nSUBSUBSECTION: {subsubsection.subsubheading}\n"
                    )

                    if subsubsection.subsubcontent:
                        section_text += subsubsection.subsubcontent + "\n"

            content.append(truncate(section_text))

    return "\n".join(content)

# ============================================================
# LLM
# ============================================================
# For local Ollama model
# from langchain_ollama import ChatOllama

# llm = ChatOllama(
#     model="qwen3:1.7b",
#     temperature=0
# )

# Using OpenCode Go (subscription lane, OpenAI-compatible) with DeepSeek V4.1 Flash

load_dotenv()

DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"

MODEL = os.getenv("OPENCODE_MODEL", "deepseek-v4.1-flash")

# The Go lane asks clients to identify themselves and to send a stable session
# id per conversation so requests can be routed and prompt-cached.
CLIENT_USER_AGENT = "paper-synthesizer/0.1"
SESSION_ID = os.getenv("OPENCODE_SESSION") or uuid.uuid4().hex

_llm = None


def default_headers() -> dict[str, str]:
    return {
        "User-Agent": CLIENT_USER_AGENT,
        "x-opencode-session": os.getenv("OPENCODE_SESSION") or SESSION_ID,
    }


def get_llm() -> ChatOpenAI:
    """Build the OpenCode Go client on first use.

    Deferred until a synthesis pass actually runs so that importing this
    module, or fetching and parsing a paper, does not require an API key.
    """
    global _llm

    if _llm is None:
        key = os.getenv("OPENCODE_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENCODE_API_KEY is not set. Copy .env.example to .env and add "
                "the API key from your OpenCode Go subscription "
                "(https://opencode.ai/console)."
            )

        _llm = ChatOpenAI(
            model=MODEL,
            base_url=os.getenv("OPENCODE_BASE_URL", DEFAULT_BASE_URL),
            api_key=key,
            temperature=0,
            timeout=120,
            max_retries=4,
            default_headers=default_headers(),
        )

    return _llm


def preflight(timeout: float = 15.0) -> dict:
    """Check the key and model access before spending money on tokens.

    A bad key or a model the workspace cannot use otherwise surfaces only
    after a paper has been downloaded, parsed, and prompted.
    """
    import httpx

    key = os.getenv("OPENCODE_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENCODE_API_KEY is not set. Copy .env.example to .env and add "
            "the API key from your OpenCode Go subscription "
            "(https://opencode.ai/console)."
        )

    base_url = os.getenv("OPENCODE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    url = f"{base_url}/models"

    try:
        response = httpx.get(
            url,
            headers={"Authorization": f"Bearer {key}", **default_headers()},
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"could not reach {url}: {exc}") from exc

    if response.status_code == 401:
        raise RuntimeError(
            "OpenCode rejected the API key (401). Check OPENCODE_API_KEY in .env."
        )
    if response.status_code == 403:
        raise RuntimeError(
            "The key is valid but model access is disabled (403). Enable "
            f"{MODEL} for your workspace at https://opencode.ai/console."
        )
    if response.status_code == 404:
        raise RuntimeError(
            f"{url} returned 404. Check OPENCODE_BASE_URL (for the Go lane it "
            f"is {DEFAULT_BASE_URL})."
        )
    if response.status_code >= 400:
        raise RuntimeError(
            f"preflight failed with HTTP {response.status_code}: "
            f"{response.text[:200]}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"{url} did not return JSON; is the base URL correct?") from exc

    models = payload.get("data", payload) if isinstance(payload, dict) else payload
    model_ids = [model.get("id") for model in models if isinstance(model, dict)]

    if MODEL not in model_ids:
        available = ", ".join(sorted(m for m in model_ids if m)[:8])
        raise RuntimeError(
            f"{MODEL} is not offered on this lane. Set OPENCODE_MODEL to one "
            f"of: {available}..."
        )

    return {"model": MODEL, "base_url": base_url, "model_count": len(model_ids)}


def structured_output(schema):
    """Wrap the client so the model answers with `schema`-shaped JSON.

    The Go lane advertises structured-output support, so this uses the
    OpenAI `response_format` json_schema path. Switch to
    method="function_calling" if a model ever rejects the strict schema.
    """
    return get_llm().with_structured_output(schema, method="json_schema")


# DeepSeek V4.1 Flash occasionally answers a long extraction prompt with
# "..." or a fragment of the instructions. A schema-valid but empty answer
# would silently corrupt the report, so each pass checks its fields.
MIN_TEXT_CHARS = 40


def _is_degenerate(value: str, min_chars: int) -> bool:
    stripped = (value or "").strip().strip(".").strip()
    return len(stripped) < min_chars


def checked_structured(schema, prompt: str, fields: tuple[str, ...], min_chars: int = MIN_TEXT_CHARS):
    """Invoke a structured-output pass and retry once on an unusable answer."""
    for _ in range(2):
        result = structured_output(schema).invoke(prompt)

        bad = [
            field
            for field in fields
            if _is_degenerate(getattr(result, field, ""), min_chars)
        ]
        if not bad:
            return result

        log.warning("model returned unusable %s; retrying", ", ".join(bad))
        prompt = (
            prompt
            + "\n\nYour previous response was empty or unusable. Answer every "
            "field fully, using specific content from the paper."
        )

    raise RuntimeError(
        "the model repeatedly returned empty or unusable output for: "
        + ", ".join(bad)
    )


# Long sections are truncated before prompting: a 60-page paper should not
# turn one request into six figures of tokens by accident.
MAX_SECTION_CHARS = int(os.getenv("PSYNTH_MAX_SECTION_CHARS", "24000"))


def set_max_section_chars(limit: int) -> None:
    global MAX_SECTION_CHARS
    MAX_SECTION_CHARS = int(limit)


def truncate(text: str, limit: int | None = None) -> str:
    """Cut text to ``limit`` characters, marking that it happened."""
    limit = MAX_SECTION_CHARS if limit is None else int(limit)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n[... truncated ...]"

# ============================================================
# Clearly Defining LLM Output Schema
# ============================================================
# For Pass 1: Overarching Synthesis
class SynthesisOutput(BaseModel):
    synthesis: str

# For Intermediary 1.5: Identifying Relevant Sections for Research Analysis
class RelevantSections(BaseModel):
    section_indices: list[int] = Field(default_factory=list)

# For Pass 2: Research Analysis
class ResearchAnalysis(BaseModel):
    hypothesis: str
    knowledge_gap: str
    results: str
    implications: str

# For Pass 3: Running Synthesis
class SectionSynthesisOutput(BaseModel):
    synthesis: str
    remarks: str
    updated_running_synthesis: str

# ============================================================
# Setting up LangGraph
# ============================================================

class GraphState(TypedDict):
    paper: PaperSynthesis
    all_sections: bool
    state_path: str | None


def save_state(state_path: str | None, paper: PaperSynthesis) -> None:
    """Persist synthesis progress so an aborted run can be resumed."""
    if not state_path:
        return

    path = Path(state_path)
    tmp = path.with_name(path.name + ".part")
    try:
        tmp.write_text(paper.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

# Pass 1 Node: Take Title, Abstract, Conclusion, and Paper Structure to generate overarching synthesis
def generate_overarching_synthesis(state: GraphState) -> GraphState:

    paper = state["paper"]

    if paper.overarching_synthesis:
        log.info("Reusing existing overarching synthesis")
        return state

    structure = get_paper_structure(paper)

    prompt = f"""
You are synthesizing a research paper.

Use the following information to develop an OVERARCHING SYNTHESIS of what the paper is fundamentally about.
Write for a knowledgeable undergraduate reader.

TITLE:
{paper.title}

ABSTRACT:
{paper.abstract}

CONCLUSION:
{paper.conclusion}

PAPER STRUCTURE:
{structure}

Focus on:
- the central research problem
- the paper's primary objective
- the general approach taken
- the main contribution
- the overall direction of the paper

Keep it concise yet comprehensive. 
Do not invent information that cannot reasonably be inferred from the provided material.
"""

    result = checked_structured(SynthesisOutput, prompt, ("synthesis",))

    paper.overarching_synthesis = result.synthesis

    return state

# Pass 1.5 Node: Identify Sections for Research Analysis on Pass 2
def identify_relevant_sections(state: GraphState) -> GraphState:

    paper = state["paper"]

    if paper.relevant_section_indices:
        log.info("Reusing section selection: %s", paper.relevant_section_indices)
        return state

    structure = get_indexed_paper_structure(paper)

    prompt = f"""
You are analyzing the structure of a research paper.

Your task is to identify the 2-4 TOP-LEVEL SECTIONS that are most likely
to contain information needed to answer the following research-analysis
questions:

1. What is the knowledge gap or research problem being addressed?
2. What is the paper's hypothesis, research question, or expected outcome?
3. What are the main results and how do the researchers interpret those results?
4. What are the implications or broader significance of the findings to the field?

Use the paper's overarching synthesis as additional context.

OVERARCHING SYNTHESIS:
{paper.overarching_synthesis}

INDEXED PAPER STRUCTURE:
{structure}

Return ONLY the indices of the 2-4 most relevant TOP-LEVEL sections.

Prioritize sections that are likely to contain substantive evidence for
the research analysis. For example, Introduction often contains the
research problem, knowledge gap, and hypothesis, while Results and
Discussion often contain findings, interpretation, and implications.

Do not assume that a section is relevant solely because of its conventional
name. Use the actual paper structure and headings.

The indices must correspond exactly to the numbered top-level sections
provided above.
"""

    result = structured_output(RelevantSections).invoke(prompt)

    paper.relevant_section_indices = result.section_indices

    return state

# Pass 2 Node: Research Analysis to Generate Knowledge Gap, Hypothesis, Results, and Implications
def generate_research_analysis(state: GraphState) -> GraphState:

    paper = state["paper"]

    if paper.hypothesis and paper.knowledge_gap and paper.results and paper.implications:
        log.info("Reusing existing research analysis")
        return state

    relevant_content = get_relevant_section_content(paper)

    prompt = f"""
You are analyzing a research paper.

Use the overarching synthesis and the selected paper sections to identify
and synthesize the following four aspects:

1. KNOWLEDGE GAP
What existing limitation, unanswered question, problem, or gap in the
literature motivates this research?

2. HYPOTHESIS
What hypothesis, research question, objective, or expected outcome does
the paper investigate?

3. RESULTS
What are the main findings of the research? Focus on the findings that
directly address the research problem or hypothesis.

4. IMPLICATIONS
How do the researchers interpret their findings, and what do those
findings mean for the broader field? Include important practical,
theoretical, or research implications where supported.

OVERARCHING SYNTHESIS:
{paper.overarching_synthesis}

SELECTED PAPER SECTIONS:
{relevant_content}

Write for a knowledgeable undergraduate reader.
Do not invent information. If something cannot reasonably be determined
from the provided sections, state that it is not clearly established.
Distinguish between what the researchers explicitly state and reasonable
interpretations of their findings.
"""

    result = checked_structured(
        ResearchAnalysis,
        prompt,
        ("knowledge_gap", "hypothesis", "results", "implications"),
    )

    paper.hypothesis = result.hypothesis
    paper.knowledge_gap = result.knowledge_gap
    paper.results = result.results
    paper.implications = result.implications

    return state

# Pass 3: Iterate through each Section, Subsection, Subsections for Synthesis and Remarks in line with previous information
def generate_running_synthesis(state: GraphState) -> GraphState:

    paper = state["paper"]

    running_synthesis = ""

    sections = select_sections(paper, state.get("all_sections", False))

    log.info(
        "Synthesizing %s of %s sections",
        len(sections),
        len(paper.sections),
    )

    total = len(sections)

    for position, section in enumerate(sections, start=1):

        if section.synthesis:
            log.info(
                "[%s/%s] Skipping %s (already synthesized)",
                position,
                total,
                section.heading,
            )
            running_synthesis = section.running_synthesis or running_synthesis
            continue

        log.info("[%s/%s] Synthesizing %s...", position, total, section.heading)

        subsections = "\n".join(
            f"SUBSECTION: {sub.subheading}\n{sub.subcontent}\n"
            + "\n".join(
                f"SUBSUBSECTION: {subsub.subsubheading}\n{subsub.subsubcontent}"
                for subsub in sub.subsubsection
            )
            for sub in section.subsection
        ).strip()

        section_body = truncate(
            f"{section.content}\n\n{subsections}".strip()
        )

        prompt = f"""
You are progressively synthesizing a research paper for a knowledgeable
undergraduate reader.

Your task is to analyze the CURRENT SECTION while maintaining a RUNNING
UNDERSTANDING of the paper.

Use the following:

OVERARCHING SYNTHESIS:
{paper.overarching_synthesis}

RESEARCH ANALYSIS:

KNOWLEDGE GAP:
{paper.knowledge_gap}

HYPOTHESIS:
{paper.hypothesis}

RESULTS:
{paper.results}

IMPLICATIONS:
{paper.implications}

PRECEDING RUNNING SYNTHESIS:
{running_synthesis}

CURRENT SECTION:
{section.heading}

{section_body}

Produce:

1. SYNTHESIS
Explain what this section contributes to understanding the paper.
Connect it to the research problem and overall argument where appropriate.

2. REMARKS
Provide concise clarification, interpretation, limitations, relationships, or important context that helps the reader understand the section.
Explain here key concepts if applicable. Do not invent information or introduce unsupported criticism.

3. UPDATED RUNNING SYNTHESIS
Update the preceding running synthesis by incorporating the current section, including relevant information from its subsections and subsubsections.
It should represent the cumulative understanding of the paper so far.

Keep all three reasonably concise but sufficiently comprehensive.
"""

        result = checked_structured(
            SectionSynthesisOutput,
            prompt,
            ("synthesis", "remarks", "updated_running_synthesis"),
            min_chars=20,
        )

        section.synthesis = result.synthesis
        section.remarks = result.remarks

        running_synthesis = result.updated_running_synthesis
        section.running_synthesis = running_synthesis
        paper.running_synthesis = running_synthesis

        save_state(state.get("state_path"), paper)

    paper.running_synthesis = running_synthesis

    return state

# ============================================================
# Building LangGraph Workflow
# ============================================================

builder = StateGraph(GraphState)

builder.add_node(
    "generate_overarching_synthesis",
    generate_overarching_synthesis
)

builder.add_node(
    "identify_relevant_sections",
    identify_relevant_sections
)

builder.add_node(
    "generate_research_analysis",
    generate_research_analysis
)

builder.add_node(
    "generate_running_synthesis",
    generate_running_synthesis
)

builder.add_edge(
    "generate_overarching_synthesis",
    "identify_relevant_sections"
)

builder.add_edge(
    "identify_relevant_sections",
    "generate_research_analysis"
)

builder.add_edge(
    "generate_research_analysis",
    "generate_running_synthesis"
)

builder.add_edge(
    START,
    "generate_overarching_synthesis"
)

builder.add_edge(
    "generate_running_synthesis",
    END
)

graph = builder.compile()

# ============================================================
# Markdown Output Function
# ============================================================
def _frontmatter_value(value) -> str:
    """Render a value as a YAML-safe scalar or block list."""
    if isinstance(value, list):
        if not value:
            return " []"
        return "\n" + "\n".join(
            f"  - {json.dumps(str(item), ensure_ascii=False)}" for item in value
        )
    if isinstance(value, bool):
        return f" {'true' if value else 'false'}"
    if isinstance(value, int):
        return f" {value}"
    return f" {json.dumps(str(value), ensure_ascii=False)}"


def save_synthesis_markdown(paper: PaperSynthesis, output_path: str) -> str:
    """Write the synthesis as Markdown with provenance frontmatter.

    Returns the path written; the caller owns the user-facing confirmation.
    """
    synthesized = [section for section in paper.sections if section.synthesis]
    skipped = [section.heading for section in paper.sections if not section.synthesis]
    arxiv_url = paper.arxiv_url or f"https://arxiv.org/abs/{paper.paper_id}"

    frontmatter = {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "authors": paper.authors,
        "arxiv_url": arxiv_url,
        "version": paper.version,
        "source": paper.source,
        "model": paper.model,
        "generated_at": paper.generated_at,
        "sections_synthesized": len(synthesized),
        "sections_available": len(paper.sections),
    }

    lines = ["---"]
    for key, value in frontmatter.items():
        lines.append(f"{key}:{_frontmatter_value(value)}")
    lines.extend(["---", ""])

    lines.append(f"# {paper.title}")
    lines.append("")

    if paper.authors:
        lines.append(f"**Authors:** {', '.join(paper.authors)}")
    lines.append(f"**Paper ID:** {paper.paper_id}")
    lines.append(f"**Source:** [{paper.paper_id}]({arxiv_url})")
    lines.append("")

    if paper.abstract:
        lines.append("## Abstract")
        lines.append("")
        lines.append(paper.abstract)
        lines.append("")

    lines.append("## Overarching Synthesis")
    lines.append("")
    lines.append(paper.overarching_synthesis)
    lines.append("")

    lines.append("## Knowledge Gap")
    lines.append("")
    lines.append(paper.knowledge_gap)
    lines.append("")

    lines.append("## Hypothesis")
    lines.append("")
    lines.append(paper.hypothesis)
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append(paper.results)
    lines.append("")

    lines.append("## Implications")
    lines.append("")
    lines.append(paper.implications)
    lines.append("")

    lines.append("## Final Running Synthesis")
    lines.append("")
    lines.append(paper.running_synthesis)
    lines.append("")

    lines.append("## Individual Section Synthesis and Remarks")
    lines.append("")

    for index, section in enumerate(paper.sections):
        if not section.synthesis:
            continue

        lines.append(f"### [{index}] {section.heading}")
        lines.append("")
        lines.append("#### Synthesis")
        lines.append("")
        lines.append(section.synthesis)
        lines.append("")
        lines.append("#### Remarks")
        lines.append("")
        lines.append(section.remarks)
        lines.append("")

    if skipped:
        lines.append("## Not Synthesized")
        lines.append("")
        lines.append(
            "Coverage was limited to the sections selected for analysis. "
            "These sections were not synthesized:"
        )
        lines.append("")
        for heading in skipped:
            lines.append(f"- {heading}")
        lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    return output_path