import json
from pydantic import BaseModel, Field
import os


# ============================================================
# Subsubsection
# ============================================================

class Subsubsection(BaseModel):
    subsubheading: str
    subsubcontent: str
    questions: list[str] = Field(default_factory=list)
    synthesis: str = ""


# ============================================================
# Subsection
# ============================================================

class Subsection(BaseModel):
    subheading: str
    subcontent: str
    questions: list[str] = Field(default_factory=list)
    synthesis: str = ""
    subsubsection: list[Subsubsection] = Field(default_factory=list)


# ============================================================
# Section
# ============================================================

class Section(BaseModel):
    heading: str
    content: str
    questions: list[str] = Field(default_factory=list)
    synthesis: str = ""
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
# Helper Function for Getting Paper Structure (Sections and Subsections)
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
# To be established/created next
# contains 2-4 section indices from Sections Object list

# For Pass 2: Research Analysis
class ResearchAnalysis(BaseModel):
    hypothesis: str
    knowledge_gap: str
    results: str
    implications: str
# to be used later after intermediary 1.5 implementation is complete

# For Pass 3: Running Synthesis
# To be established/created later after pass 2 implementation is complete

# ============================================================
# Setting up LangGraph
# ============================================================
from typing import TypedDict

class GraphState(TypedDict):
    paper: PaperSynthesis

# First Pass Node: Take Title, Abstract, Conclusion, and Paper Structure to generate overarching synthesis
def generate_overarching_synthesis(state: GraphState) -> GraphState:

    paper = state["paper"]

    structure = get_paper_structure(paper)

    prompt = f"""
You are synthesizing a research paper to be understood by a knowledgeable undergrad student of the field.

Use the following information to develop an OVERARCHING SYNTHESIS of what the paper is fundamentally about.

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


# ============================================================
# Building LangGraph Workflow
# ============================================================

from langgraph.graph import StateGraph, START, END

builder = StateGraph(GraphState)

builder.add_node(
    "generate_overarching_synthesis",
    generate_overarching_synthesis
)

builder.add_edge(
    START,
    "generate_overarching_synthesis"
)

builder.add_edge(
    "generate_overarching_synthesis",
    END
)

graph = builder.compile()

# ============================================================
# Main Usage
# ============================================================

paper_id = "2608.27454"
json_path = f"{paper_id}/{paper_id}.json"

paper = PaperSynthesis.from_json(json_path)

# print(paper.paper_id)
# print(paper.title)
# print(paper.authors)
# print(paper.abstract)
# print(paper.conclusion)

print(f"Number of sections: {len(paper.sections)}")

# print("Paper Structure:")
# print(get_paper_structure(paper))

result = graph.invoke({
    "paper": paper
})

paper = result["paper"]

print(paper.overarching_synthesis)


# Second Pass: Take Introduction, Discussion and overarching synthesis to generate primary hypothesis, knowledge gap being addressed, what the results mean in the broader context of the field
# Synthesize Introduction (Background) and Discussion (of results) Sections in line with overarching synthesis
# Generate guide questions to help synthesize other sections, contributing to the overarching picture

# Third Pass: Running Synthesis and Critiquing