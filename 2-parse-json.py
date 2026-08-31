import re
import json

# Define ArXiv paper ID
paper_id = "2608.27454"
# paper_id = "2608.26247"
# paper_id = "2608.21267"

folder_dir = f"{paper_id}"  # Folder containing the files

# Define TeX path to parse
tex_dir = f"{folder_dir}/{paper_id}.tex"  # in this case, same as paper id


def parse_latex_to_json(latex, paper_id):
    # ---------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------

    def clean_text(text):
        """Clean unnecessary whitespace while preserving paragraphs."""
        text = re.sub(r'%.*', '', text)              # Remove LaTeX comments
        text = re.sub(r'\n\s*\n+', '\n\n', text)    # Normalize paragraphs
        text = re.sub(r'[ \t]+', ' ', text)          # Normalize spaces
        return text.strip()

    def extract_braced(text, command):
        """Extract the contents of a LaTeX command such as \\title{...}."""
        pattern = rf'\\{command}\*?\s*\{{'
        match = re.search(pattern, text)

        if not match:
            return ""

        start = match.end()
        depth = 1
        i = start

        while i < len(text) and depth > 0:
            if text[i] == '{' and (i == 0 or text[i - 1] != '\\'):
                depth += 1
            elif text[i] == '}' and (i == 0 or text[i - 1] != '\\'):
                depth -= 1

            i += 1

        return clean_text(text[start:i - 1])

    def extract_authors(text):
        """Extract all \\author{...} commands."""
        authors = []

        pattern = r'\\author\*?\s*(?:\[[^\]]*\])?\s*\{'
        matches = list(re.finditer(pattern, text))

        for match in matches:
            start = match.end()
            depth = 1
            i = start

            while i < len(text) and depth > 0:
                if text[i] == '{' and (i == 0 or text[i - 1] != '\\'):
                    depth += 1
                elif text[i] == '}' and (i == 0 or text[i - 1] != '\\'):
                    depth -= 1

                i += 1

            author = clean_text(text[start:i - 1])

            if author:
                authors.append(author)

        return authors

    # ---------------------------------------------------------
    # Extract document metadata
    # ---------------------------------------------------------

    title = extract_braced(latex, "title")
    authors = extract_authors(latex)
    abstract_match = re.search(
        r'\\begin\s*\{\s*abstract\s*\}(.*?)\\end\s*\{\s*abstract\s*\}',
        latex,
        re.DOTALL
    )

    abstract = clean_text(abstract_match.group(1)) if abstract_match else ""

    # ---------------------------------------------------------
    # Only parse the actual document body
    # ---------------------------------------------------------

    document_match = re.search(
        r'\\begin\s*\{\s*document\s*\}',
        latex
    )

    if document_match:
        body = latex[document_match.end():]
    else:
        body = latex

    # Stop parsing at bibliography or end of document
    stop_match = re.search(
        r'\\(?:bibliography|bibliographystyle|end\s*\{\s*document\s*\})',
        body
    )

    if stop_match:
        body = body[:stop_match.start()]

    # ---------------------------------------------------------
    # Find all section-level commands
    # ---------------------------------------------------------

    heading_pattern = re.compile(
        r'\\(section|subsection|subsubsection)\*?\s*\{'
    )

    matches = list(heading_pattern.finditer(body))

    sections = []
    current_section = None
    current_subsection = None

    # ---------------------------------------------------------
    # Parse the hierarchy
    # ---------------------------------------------------------

    for index, match in enumerate(matches):

        level = match.group(1)

        # Extract heading
        start = match.end()
        depth = 1
        i = start

        while i < len(body) and depth > 0:
            if body[i] == '{' and (i == 0 or body[i - 1] != '\\'):
                depth += 1
            elif body[i] == '}' and (i == 0 or body[i - 1] != '\\'):
                depth -= 1

            i += 1

        heading = clean_text(body[start:i - 1])

        # Content extends until the next heading
        content_start = i

        if index + 1 < len(matches):
            content_end = matches[index + 1].start()
        else:
            content_end = len(body)

        content = clean_text(
            body[content_start:content_end]
        )

        # -----------------------------------------------------
        # SECTION
        # -----------------------------------------------------

        if level == "section":

            current_section = {
                "heading": heading,
                "content": content,
                "subsection": []
            }

            sections.append(current_section)
            current_subsection = None

        # -----------------------------------------------------
        # SUBSECTION
        # -----------------------------------------------------

        elif level == "subsection":

            # Ignore malformed subsections appearing before a section
            if current_section is None:
                continue

            current_subsection = {
                "subheading": heading,
                "subcontent": content,
                "subsubsection": []
            }

            current_section["subsection"].append(
                current_subsection
            )

        # -----------------------------------------------------
        # SUBSUBSECTION
        # -----------------------------------------------------

        elif level == "subsubsection":

            # Ignore malformed subsubsections without a subsection
            if current_subsection is None:
                continue

            subsubsection = {
                "subsubheading": heading,
                "subsubcontent": content
            }

            current_subsection["subsubsection"].append(
                subsubsection
            )

    # ---------------------------------------------------------
    # Remove section content that actually belongs to
    # subsections/subsubsections.
    #
    # The loop above initially assigns content up to the next
    # heading. Recalculate each level's content so that:
    #
    # section content     = before first subsection
    # subsection content  = before first subsubsection
    # subsubsection       = until next heading
    # ---------------------------------------------------------

    for section in sections:

        # Locate subsection headings in the original structure
        # using the already parsed hierarchy.
        if section["subsection"]:

            # Section content was originally allowed to include
            # subsection text, so it is already necessary to
            # derive it from the source. This is handled below.
            pass

    # ---------------------------------------------------------
    # Re-parse content boundaries precisely
    # ---------------------------------------------------------

    sections = []

    matches = list(heading_pattern.finditer(body))

    for index, match in enumerate(matches):

        if match.group(1) != "section":
            continue

        # Get section heading
        start = match.end()
        depth = 1
        i = start

        while i < len(body) and depth > 0:
            if body[i] == '{' and (i == 0 or body[i - 1] != '\\'):
                depth += 1
            elif body[i] == '}' and (i == 0 or body[i - 1] != '\\'):
                depth -= 1
            i += 1

        heading = clean_text(body[start:i - 1])

        section_start = i

        # Section ends at next section or end of body
        section_end = len(body)

        for next_match in matches[index + 1:]:
            if next_match.group(1) == "section":
                section_end = next_match.start()
                break

        section_body = body[section_start:section_end]

        # -----------------------------------------------------
        # Find subsections belonging to this section
        # -----------------------------------------------------

        subsection_matches = list(
            re.finditer(
                r'\\subsection\s*\{',
                section_body
            )
        )

        section_content = ""

        if subsection_matches:
            section_content = clean_text(
                section_body[:subsection_matches[0].start()]
            )
        else:
            section_content = clean_text(section_body)

        section_data = {
            "heading": heading,
            "content": section_content,
            "questions": [],
            "synthesis": "",
            "subsection": []
        }

        # -----------------------------------------------------
        # Parse each subsection
        # -----------------------------------------------------

        for sub_index, sub_match in enumerate(subsection_matches):

            start = sub_match.end()
            depth = 1
            i = start

            while i < len(section_body) and depth > 0:
                if section_body[i] == '{' and (
                    i == 0 or section_body[i - 1] != '\\'
                ):
                    depth += 1
                elif section_body[i] == '}' and (
                    i == 0 or section_body[i - 1] != '\\'
                ):
                    depth -= 1
                i += 1

            subheading = clean_text(
                section_body[start:i - 1]
            )

            subsection_start = i

            # Find next subsection
            subsection_end = len(section_body)

            if sub_index + 1 < len(subsection_matches):
                subsection_end = subsection_matches[
                    sub_index + 1
                ].start()

            subsection_body = section_body[
                subsection_start:subsection_end
            ]

            # -------------------------------------------------
            # Find subsubsections
            # -------------------------------------------------

            subsub_matches = list(
                re.finditer(
                    r'\\subsubsection\s*\{',
                    subsection_body
                )
            )

            if subsub_matches:
                subcontent = clean_text(
                    subsection_body[:subsub_matches[0].start()]
                )
            else:
                subcontent = clean_text(subsection_body)

            subsection_data = {
                "subheading": subheading,
                "subcontent": subcontent,
                "questions": [],
                "synthesis": "",
                "subsubsection": []
            }

            # -------------------------------------------------
            # Parse subsubsections
            # -------------------------------------------------

            for subsub_index, subsub_match in enumerate(
                subsub_matches
            ):

                start = subsub_match.end()
                depth = 1
                i = start

                while i < len(subsection_body) and depth > 0:
                    if subsection_body[i] == '{' and (
                        i == 0 or subsection_body[i - 1] != '\\'
                    ):
                        depth += 1
                    elif subsection_body[i] == '}' and (
                        i == 0 or subsection_body[i - 1] != '\\'
                    ):
                        depth -= 1

                    i += 1

                subsubheading = clean_text(
                    subsection_body[start:i - 1]
                )

                subsubcontent_start = i

                if subsub_index + 1 < len(subsub_matches):
                    subsubcontent_end = subsub_matches[
                        subsub_index + 1
                    ].start()
                else:
                    subsubcontent_end = len(subsection_body)

                subsubcontent = clean_text(
                    subsection_body[
                        subsubcontent_start:subsubcontent_end
                    ]
                )

                subsection_data["subsubsection"].append({
                    "subsubheading": subsubheading,
                    "subsubcontent": subsubcontent,
                    "questions": [],
                    "synthesis": ""
                })

            section_data["subsection"].append(
                subsection_data
            )

        sections.append(section_data)

    # ---------------------------------------------------------
    # Build final JSON structure
    # ---------------------------------------------------------

    result = {
        "paper_id": paper_id,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "section": sections
    }

    return result


# usage

with open(tex_dir, "r", encoding="utf-8") as f:
    latex = f.read()

parsed = parse_latex_to_json(latex, paper_id)

with open(f"{folder_dir}/{paper_id}.json", "w", encoding="utf-8") as f:
    json.dump(parsed, f, indent=2, ensure_ascii=False)

print(f"Saved {paper_id}.json")