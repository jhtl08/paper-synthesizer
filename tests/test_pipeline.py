import json

import pytest

import synthesistools as s


class Stub:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def make_paper(tmp_path, count=5):
    data = {
        "paper_id": "test.0001",
        "title": "Test Paper",
        "authors": ["Ada"],
        "abstract": "An abstract.",
        "conclusion": "Done.",
        "section": [
            {"heading": f"S{i}", "content": f"body {i}", "subsection": []}
            for i in range(count)
        ],
    }
    path = tmp_path / "paper.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return s.PaperSynthesis.from_json(str(path))


def install_stub(monkeypatch, selected=(0, 1), fail_at=None):
    calls = []

    def fake_structured(schema):
        name = schema.__name__

        class Runner:
            def invoke(self, prompt):
                calls.append((name, prompt))
                if name == "SynthesisOutput":
                    return Stub(
                        synthesis="A substantive overview of what the paper is about overall."
                    )
                if name == "RelevantSections":
                    return Stub(section_indices=list(selected))
                if name == "ResearchAnalysis":
                    return Stub(
                        hypothesis=(
                            "A substantive hypothesis that the paper investigates in depth."
                        ),
                        knowledge_gap=(
                            "A substantive knowledge gap that motivates the research problem."
                        ),
                        results=(
                            "A substantive summary of the main empirical results obtained."
                        ),
                        implications=(
                            "Substantive implications of these findings for the broader field."
                        ),
                    )
                completed = sum(1 for call in calls if call[0] == "SectionSynthesisOutput")
                if fail_at is not None and completed == fail_at:
                    raise RuntimeError("simulated failure")
                return Stub(
                    synthesis=f"A substantive synthesis of section {completed}.",
                    remarks="A substantive remark about this section.",
                    updated_running_synthesis=f"A substantive running synthesis {completed}.",
                )

        return Runner()

    monkeypatch.setattr(s, "structured_output", fake_structured)
    return calls


def test_pass_three_only_synthesizes_selected_sections(tmp_path, monkeypatch):
    install_stub(monkeypatch, selected=(1, 3))
    paper = make_paper(tmp_path)

    result = s.graph.invoke(
        {"paper": paper, "all_sections": False, "state_path": None}
    )

    synthesized = [section.heading for section in result["paper"].sections if section.synthesis]
    assert synthesized == ["S1", "S3"]


def test_all_sections_flag_overrides_selection(tmp_path, monkeypatch):
    install_stub(monkeypatch, selected=(1, 3))
    paper = make_paper(tmp_path)

    result = s.graph.invoke(
        {"paper": paper, "all_sections": True, "state_path": None}
    )

    assert all(section.synthesis for section in result["paper"].sections)


def test_out_of_range_selection_falls_back_to_all(tmp_path):
    paper = make_paper(tmp_path)
    paper.relevant_section_indices = [99]

    assert len(s.select_sections(paper)) == len(paper.sections)


def test_truncation_reaches_pass_three_prompt(tmp_path, monkeypatch):
    calls = install_stub(monkeypatch)
    s.set_max_section_chars(3)
    try:
        paper = make_paper(tmp_path)
        s.graph.invoke({"paper": paper, "all_sections": False, "state_path": None})
    finally:
        s.set_max_section_chars(24000)

    pass_three = [prompt for name, prompt in calls if name == "SectionSynthesisOutput"]
    assert pass_three
    assert "[... truncated ...]" in pass_three[0]


def test_resume_skips_completed_work(tmp_path, monkeypatch):
    paper = make_paper(tmp_path)
    paper.overarching_synthesis = "overview"
    paper.relevant_section_indices = [0, 1]
    paper.hypothesis = paper.knowledge_gap = paper.results = paper.implications = "done"
    paper.sections[0].synthesis = "already"
    paper.sections[0].running_synthesis = "run-0"

    calls = install_stub(monkeypatch, selected=(0, 1))
    result = s.graph.invoke(
        {"paper": paper, "all_sections": False, "state_path": None}
    )

    assert [name for name, _ in calls] == ["SectionSynthesisOutput"]
    assert "PRECEDING RUNNING SYNTHESIS:\nrun-0" in calls[0][1]
    assert result["paper"].sections[1].synthesis


def test_degenerate_output_is_retried(monkeypatch):
    calls = {"count": 0}

    def fake_structured(schema):
        class Runner:
            def invoke(self, prompt):
                calls["count"] += 1
                if calls["count"] == 1:
                    return Stub(synthesis="...", remarks=".", updated_running_synthesis="")
                return Stub(
                    synthesis="A substantive synthesis of this section.",
                    remarks="A substantive remark about the content.",
                    updated_running_synthesis="A substantive running synthesis so far.",
                )

        return Runner()

    monkeypatch.setattr(s, "structured_output", fake_structured)

    result = s.checked_structured(
        s.SectionSynthesisOutput,
        "prompt",
        ("synthesis", "remarks", "updated_running_synthesis"),
        min_chars=20,
    )

    assert calls["count"] == 2
    assert result.synthesis.startswith("A substantive")


def test_persistently_degenerate_output_raises(monkeypatch):
    class Runner:
        def invoke(self, prompt):
            return Stub(hypothesis="...", knowledge_gap="", results=".", implications=" ")

    monkeypatch.setattr(s, "structured_output", lambda schema: Runner())

    with pytest.raises(RuntimeError, match="repeatedly"):
        s.checked_structured(
            s.ResearchAnalysis,
            "prompt",
            ("knowledge_gap", "hypothesis", "results", "implications"),
        )


def test_state_is_written_after_each_section(tmp_path, monkeypatch):
    install_stub(monkeypatch, selected=(0, 1, 2))
    paper = make_paper(tmp_path)
    state_path = tmp_path / "state.json"

    s.graph.invoke(
        {"paper": paper, "all_sections": False, "state_path": str(state_path)}
    )

    saved = s.PaperSynthesis.model_validate_json(state_path.read_text(encoding="utf-8"))
    assert sum(1 for section in saved.sections if section.synthesis) == 3
