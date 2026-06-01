import re
import json
import pandas as pd
from pypdf import PdfReader


def extract_chapter_number(title: str) -> int | None:
    """
    Extract the chapter number from a chapter title.

    Supports multiple chapter title formats such as:
    - "Chapter 1"
    - "Chapter 10: Introduction"
    - "Ch 3"
    - "3 Getting Started"

    Args:
        title (str): Raw title extracted from the PDF outline.

    Returns:
        int | None: Extracted chapter number if found, otherwise None.
    """

    patterns = [r"Chapter\s+(\d+)", r"Ch(?:apter)?\.?\s*(\d+)", r"^(\d+)"]

    for pattern in patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def clean_title(title: str) -> str:
    """
    Clean chapter, section, or subsection titles.

    Removes chapter prefixes and hierarchical numbering from
    PDF outline titles.

    Examples:
        "Chapter 3: Data Processing" -> "Data Processing"
        "2.1 Feature Engineering" -> "Feature Engineering"

    Args:
        title (str): Original title from the PDF outline.

    Returns:
        str: Cleaned title.
    """

    title = re.sub(r"Chapter\s+\d+[:.\-]?\s*", "", title, flags=re.IGNORECASE)

    title = re.sub(r"^\d+(\.\d+)*\s*", "", title)

    return title.strip()


def build_hierarchy(chapter_num, chapter, section=None, subsection=None) -> str:
    """
    Build a hierarchy path string.

    Creates a hierarchical representation of a document section.

    Example:
        Ch3 > Data Processing > Feature Engineering > Scaling

    Args:
        chapter_num (int | None): Chapter number.
        chapter (str | None): Chapter title.
        section (str | None): Section title.
        subsection (str | None): Subsection title.

    Returns:
        str: Hierarchical path.
    """

    parts = []

    if chapter_num is not None:
        parts.append(f"Ch{chapter_num}")

    if chapter:
        parts.append(chapter)

    if section:
        parts.append(section)

    if subsection:
        parts.append(subsection)

    return " > ".join(parts)


def parse_outline(reader, outline, level=1, state=None, structured_data=None) -> list[dict]:
    """
    Parse a PDF outline recursively.

    Traverses the PDF table of contents and extracts chapter,
    section, and subsection metadata.

    Args:
        reader (PdfReader): PDF reader instance.
        outline (list): PDF outline tree.
        level (int, optional): Current outline depth level.
            Defaults to 1.
        state (dict | None, optional): Current hierarchy state.
            Defaults to None.
        structured_data (list | None, optional): Accumulated
            metadata records. Defaults to None.

    Returns:
        list[dict]: Extracted document structure metadata.
    """

    if state is None:
        state = {"chapter": None, "chapter_number": None, "section": None}

    if structured_data is None:
        structured_data = []

    for item in outline:

        if isinstance(item, list):

            parse_outline(reader, item, level + 1, state, structured_data)

            continue

        title = item.title.strip()
        clean = clean_title(title)

        page_number = reader.get_destination_page_number(item) + 1

        metadata = {
            "chapter_number": None,
            "chapter_title": None,
            "section_title": None,
            "subsection_title": None,
            "hierarchy_path": None,
            "page_number": page_number,
            "level": level,
        }

        if level == 1:

            state["chapter"] = clean
            state["chapter_number"] = extract_chapter_number(title)
            state["section"] = None

            metadata.update(
                {
                    "chapter_number": state["chapter_number"],
                    "chapter_title": state["chapter"],
                    "hierarchy_path": build_hierarchy(
                        state["chapter_number"], state["chapter"]
                    ),
                }
            )

        elif level == 2:

            state["section"] = clean

            metadata.update(
                {
                    "chapter_number": state["chapter_number"],
                    "chapter_title": state["chapter"],
                    "section_title": state["section"],
                    "hierarchy_path": build_hierarchy(
                        state["chapter_number"], state["chapter"], state["section"]
                    ),
                }
            )

        else:

            metadata.update(
                {
                    "chapter_number": state["chapter_number"],
                    "chapter_title": state["chapter"],
                    "section_title": state["section"],
                    "subsection_title": clean,
                    "hierarchy_path": build_hierarchy(
                        state["chapter_number"],
                        state["chapter"],
                        state["section"],
                        clean,
                    ),
                }
            )

        structured_data.append(metadata)

    return structured_data


def clean_dataframe(df) -> pd.DataFrame:
    """
    Clean and normalize the structure DataFrame.

    Performs:
    - Null handling
    - String normalization
    - Chapter number conversion

    Args:
        df (pd.DataFrame): Raw structure DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """

    text_cols = ["chapter_title", "section_title", "subsection_title", "hierarchy_path"]

    for col in text_cols:

        df[col] = df[col].fillna("").astype(str).str.replace(". ", "", regex=False)

        df[col] = df[col].replace("", None)

    df["chapter_number"] = df["chapter_number"].apply(
        lambda x: None if pd.isna(x) else int(x)
    )

    return df


def save_json(data, path) -> None:
    """
    Save data to a JSON file.

    Args:
        data (list | dict): Data to save.
        path (str): Output JSON file path.

    Returns:
        None
    """

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_json_to_dataframe(path) -> pd.DataFrame:
    """
    Load a JSON file into a pandas DataFrame.

    Args:
        path (str): Path to the JSON file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """

    return pd.read_json(path)


def build_pdf_structure(pdf_path: str, output_json: str) -> pd.DataFrame:
    """
    Build a structured representation of a PDF outline.

    This pipeline:
    1. Loads the PDF.
    2. Parses the outline hierarchy.
    3. Saves the extracted structure as JSON.
    4. Loads and cleans the structure.
    5. Saves the cleaned structure.

    Args:
        pdf_path (str): Path to the PDF file.
        output_json (str): Output JSON file path.

    Returns:
        pd.DataFrame: Cleaned document structure DataFrame.
    """

    reader = PdfReader(pdf_path)

    structured_data = parse_outline(reader=reader, outline=reader.outline)

    save_json(structured_data, output_json)

    df = load_json_to_dataframe(output_json).sort_values("page_number")

    df = clean_dataframe(df)

    df.to_json(output_json, orient="records", force_ascii=False, indent=4)

    return df
