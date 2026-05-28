from typing import Any

from langchain_core.documents import Document

import pandas as pd
import numpy as np


def load_structure_dataframe(json_path: str) -> pd.DataFrame:
    """
    Load and sort PDF structure metadata.

    Args:
        json_path (str): Path to the structure JSON file.

    Returns:
        pd.DataFrame: Sorted structure metadata.
    """

    return pd.read_json(json_path).sort_values("page_number").reset_index(drop=True)


def deep_clean(obj: Any) -> Any:
    """
    Recursively clean Python objects.

    Converts:
    - NumPy scalars to native Python types.
    - NaN values to None.
    - Nested dictionaries and lists recursively.

    Args:
        obj (Any): Object to clean.

    Returns:
        Any: Cleaned object.
    """

    if isinstance(obj, dict):
        return {k: deep_clean(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [deep_clean(v) for v in obj]

    if isinstance(obj, np.generic):
        obj = obj.item()

    if pd.isna(obj):
        return None

    return obj


def get_structure_metadata(
    page_number: int, structure_df: pd.DataFrame
) -> dict[str, Any]:
    """
    Retrieve structure metadata for a page.

    Finds the most recent chapter, section, and subsection
    corresponding to the given page number.

    Args:
        page_number (int): PDF page number.
        structure_df (pd.DataFrame): Structure metadata DataFrame.

    Returns:
        dict[str, Any]: Structure metadata.
    """

    filtered = structure_df[structure_df["page_number"] <= page_number]

    if filtered.empty:
        return {}

    latest = filtered.iloc[-1]

    return deep_clean(
        {
            "chapter_number": latest["chapter_number"],
            "chapter_title": latest["chapter_title"],
            "section_title": latest["section_title"],
            "subsection_title": latest["subsection_title"],
            "hierarchy_path": latest["hierarchy_path"],
            "level": latest["level"],
        }
    )


def enrich_chunk_metadata(chunk: Document, structure_df: pd.DataFrame) -> Document:
    """
    Enrich a document chunk with structure metadata.

    Adds chapter, section, subsection, hierarchy path,
    and outline level information to the chunk metadata.

    Args:
        chunk (Document): Document chunk.
        structure_df (pd.DataFrame): Structure metadata DataFrame.

    Returns:
        Document: Updated document chunk.
    """

    chunk.metadata = deep_clean(chunk.metadata)

    page_number = int(chunk.metadata.get("page_label", 0))

    structure_metadata = get_structure_metadata(page_number, structure_df)

    chunk.metadata.update(structure_metadata)

    return chunk


def enrich_chunks(
    chunks: list[Document],
    structure_df: pd.DataFrame,
) -> list[Document]:
    """
    Enrich all chunks with structure metadata.

    Args:
        chunks (list[Document]): Chunked documents.
        structure_df (pd.DataFrame): Structure metadata DataFrame.

    Returns:
        list[Document]: Enriched document chunks.
    """

    return [enrich_chunk_metadata(chunk, structure_df) for chunk in chunks]
