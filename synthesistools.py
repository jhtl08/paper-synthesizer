import json
from pydantic import BaseModel, Field
import os


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

        with open(json_path, "r", encoding="utf-8") as f:
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
            sections=data.get("section", [])
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

            content.append(section_text)

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

# Using Gemini API Key
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

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
from typing import TypedDict

class GraphState(TypedDict):
    paper: PaperSynthesis

# Pass 1 Node: Take Title, Abstract, Conclusion, and Paper Structure to generate overarching synthesis
def generate_overarching_synthesis(state: GraphState) -> GraphState:

    paper = state["paper"]

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

    result = llm.with_structured_output(SynthesisOutput).invoke(prompt)

    paper.overarching_synthesis = result.synthesis

    return state

# Pass 1.5 Node: Identify Sections for Research Analysis on Pass 2
def identify_relevant_sections(state: GraphState) -> GraphState:

    paper = state["paper"]

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

    result = llm.with_structured_output(RelevantSections).invoke(prompt)

    paper.relevant_section_indices = result.section_indices

    return state

# Pass 2 Node: Research Analysis to Generate Knowledge Gap, Hypothesis, Results, and Implications
def generate_research_analysis(state: GraphState) -> GraphState:

    paper = state["paper"]

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

    result = llm.with_structured_output(ResearchAnalysis).invoke(prompt)

    paper.hypothesis = result.hypothesis
    paper.knowledge_gap = result.knowledge_gap
    paper.results = result.results
    paper.implications = result.implications

    return state

# Pass 3: Iterate through each Section, Subsection, Subsections for Synthesis and Remarks in line with previous information
def generate_running_synthesis(state: GraphState) -> GraphState:

    paper = state["paper"]

    running_synthesis = ""

    for section in paper.sections:

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

{section.content}

SUBSECTIONS AND SUBSUBSECTIONS:
{chr(10).join(
    f"""
SUBSECTION: {sub.subheading}
{sub.subcontent}

{chr(10).join(
    f"SUBSUBSECTION: {subsub.subsubheading}\n{subsub.subsubcontent}"
    for subsub in sub.subsubsection
)}
"""
    for sub in section.subsection
)}

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

        result = llm.with_structured_output(
            SectionSynthesisOutput
        ).invoke(prompt)

        section.synthesis = result.synthesis
        section.remarks = result.remarks

        running_synthesis = result.updated_running_synthesis

    paper.running_synthesis = running_synthesis

    return state

# ============================================================
# Building LangGraph Workflow
# ============================================================

from langgraph.graph import StateGraph, START, END

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
def save_synthesis_markdown(paper: PaperSynthesis, output_path: str):
    """Save the complete paper synthesis as a structured Markdown file."""

    with open(output_path, "w", encoding="utf-8") as f:

        # ====================================================
        # Title
        # ====================================================

        f.write(f"# {paper.title}\n\n")
        f.write(f"**Paper ID:** {paper.paper_id}\n\n")

        # ====================================================
        # Pass 1
        # ====================================================

        f.write("## Overarching Synthesis\n\n")
        f.write(f"{paper.overarching_synthesis}\n\n")

        # ====================================================
        # Pass 2
        # ====================================================

        f.write("## Knowledge Gap\n\n")
        f.write(f"{paper.knowledge_gap}\n\n")

        f.write("## Hypothesis\n\n")
        f.write(f"{paper.hypothesis}\n\n")

        f.write("## Results\n\n")
        f.write(f"{paper.results}\n\n")

        f.write("## Implications\n\n")
        f.write(f"{paper.implications}\n\n")

        # ====================================================
        # Pass 3 — Final Running Synthesis
        # ====================================================

        f.write("## Final Running Synthesis\n\n")
        f.write(f"{paper.running_synthesis}\n\n")

        # ====================================================
        # Pass 3 — Individual Sections
        # ====================================================

        f.write("## Individual Section Synthesis and Remarks\n\n")

        for i, section in enumerate(paper.sections):

            f.write(f"### [{i}] {section.heading}\n\n")

            f.write("#### Synthesis\n\n")
            f.write(f"{section.synthesis}\n\n")

            f.write("#### Remarks\n\n")
            f.write(f"{section.remarks}\n\n")

    print(f"Synthesis saved to: {output_path}")