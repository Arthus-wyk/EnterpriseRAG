import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import CONFIG


@dataclass(frozen=True)
class Chunk:
    text: str
    metadata: dict[str, Any]

    @property
    def source(self) -> str:
        return str(self.metadata.get("source_path", ""))

    @property
    def index(self) -> int:
        return int(self.metadata.get("chunk_index", 0))

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", ""))


SENTENCE_PUNCTUATIONS = [".", "。", "!", "！", "?", "？"]
SOFT_PUNCTUATIONS = [",", "，", ";", "；", ":", "："]

def extract_source_relative_path(source: str) -> str:
    path = Path(source)
    try:
        return path.relative_to(CONFIG.raw_data_dir).as_posix()
    except ValueError:
        parts = path.parts
        if "all_documents" not in parts:
            return path.name

        index = parts.index("all_documents")
        return Path(*parts[index + 1:]).as_posix()


def extract_source_type(source: str) -> str:
    relative_path = extract_source_relative_path(source)
    parts = Path(relative_path).parts
    return parts[0] if parts else "unknown"


def extract_folder_metadata(source: str) -> dict[str, Any]:
    relative_path = extract_source_relative_path(source)
    parts = Path(relative_path).parts
    folder_parts = parts[:-1]
    metadata: dict[str, Any] = {
        "folder_path": Path(*folder_parts).as_posix() if folder_parts else "",
        "folder_depth": len(folder_parts),
    }

    for index, part in enumerate(folder_parts[:5], start=1):
        metadata[f"folder_l{index}"] = part

    return metadata


def make_doc_id(source: str) -> str:
    stem = Path(source).stem
    if "__pr" in stem:
        return stem.split("__pr", 1)[0]

    relative_path = extract_source_relative_path(source)
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:16]


def make_chunk_metadata(source: str, title: str, chunk_index: int) -> dict[str, Any]:
    doc_id = make_doc_id(source)
    source_relative_path = extract_source_relative_path(source)
    chunk_id = f"{doc_id}_chunk_{chunk_index:04d}"

    metadata = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "source_path": source,
        "source_relative_path": source_relative_path,
        "source_type": extract_source_type(source),
        "title": title,
        "chunk_index": chunk_index,
    }
    metadata.update(extract_folder_metadata(source))
    return metadata


def find_cut_position(
        text: str,
        chunk_size: int,
        min_ratio: float = 2 / 3,
) -> int:
    """
    在 text 的前 chunk_size 个字符内找切分点。

    优先按句号/问号/感叹号切。
    如果切出来太短，小于 chunk_size * 2/3，
    就尝试按逗号/分号/冒号等软标点切。
    如果还是找不到，就硬切 chunk_size。
    """

    window = text[:chunk_size]
    min_size = int(chunk_size * min_ratio)

    # 1. 先找句子级标点
    sentence_cut = -1
    for p in SENTENCE_PUNCTUATIONS:
        pos = window.rfind(p)
        if pos > sentence_cut:
            sentence_cut = pos

    # 找到句子标点，并且切出来不太短
    if sentence_cut != -1 and sentence_cut + 1 >= min_size:
        return sentence_cut + 1

    # 2. 如果句子级切分太短，就找软标点
    soft_cut = -1
    for p in SOFT_PUNCTUATIONS:
        pos = window.rfind(p)
        if pos > soft_cut:
            soft_cut = pos

    if soft_cut != -1 and soft_cut + 1 >= min_size:
        return soft_cut + 1

    # 3. 如果句子标点虽然小于 2/3，但至少存在，也可以退而求其次使用
    if sentence_cut != -1:
        return sentence_cut + 1

    # 4. 如果软标点存在，也可以退而求其次使用
    if soft_cut != -1:
        return soft_cut + 1

    # 5. 实在没有标点，就硬切
    return chunk_size


def split_long_line(
        line: str,
        chunk_size: int,
        min_ratio: float = 2 / 3,
) -> list[str]:
    """
    把超过 chunk_size 的单行切成多个片段。
    优先按完整句子切，其次按逗号等标点切，最后硬切。
    """

    parts = []
    line = line.strip()

    while len(line) > chunk_size:
        cut_pos = find_cut_position(
            text=line,
            chunk_size=chunk_size,
            min_ratio=min_ratio,
        )

        part = line[:cut_pos].strip()
        if part:
            parts.append(part)

        line = line[cut_pos:].strip()

    if line:
        parts.append(line)

    return parts


def find_overlap(
        chunk: str,
        chunk_overlap: int
) -> str:
    if chunk_overlap <= 0:
        return ""

    cut = -1
    for p in SOFT_PUNCTUATIONS + SENTENCE_PUNCTUATIONS:
        pos = chunk.rfind(p)
        if pos > cut:
            cut = pos

    if cut != -1:
        tail = chunk[cut + 1:].strip()
        if tail and len(tail) <= chunk_overlap:
            return tail

    chunk = chunk.strip()
    if len(chunk) <= chunk_overlap:
        return chunk.strip()

    if chunk_overlap <= 3:
        return chunk[-chunk_overlap:].strip()

    return "..." + chunk[-(chunk_overlap - 3):].strip()


def joined_length(parts: list[str]) -> int:
    if not parts:
        return 0

    return sum(len(part) for part in parts) + len(parts) - 1


def joined_fits(parts: list[str], chunk_size: int) -> bool:
    return joined_length(parts) <= chunk_size


def merge_units_to_chunks(
        units: list[str],
        chunk_size: int,
        chunk_overlap: int
) -> list[str]:
    """
    把已经处理好的短行/短片段合并成 chunk。
    """

    chunks: list[str] = []
    current_parts: list[str] = []

    for unit in units:
        candidate = current_parts + [unit]
        if joined_fits(candidate, chunk_size):
            current_parts.append(unit)
            continue

        if current_parts:
            current_text = "\n".join(current_parts)
            chunks.append(current_text)
            overlap_content = find_overlap(current_text, chunk_overlap)
        else:
            overlap_content = ""

        next_parts = [unit]
        if overlap_content and joined_fits([overlap_content, unit], chunk_size):
            next_parts = [overlap_content, unit]

        current_parts = next_parts

    if current_parts:
        chunks.append("\n".join(current_parts))

    return chunks


def split_documents(
        documents,
        chunk_size: int,
        chunk_overlap: int
) -> list[Chunk]:
    chunks = []

    for document in documents:
        chunk_index = 0

        lines = [
            line.strip()
            for line in document.text.splitlines()
            if line.strip()
        ]

        if not lines:
            continue

        title = lines[0]
        content_lines = lines[1:]

        units = []

        for line in content_lines:
            if len(line) > chunk_size:
                parts = split_long_line(
                    line=line,
                    chunk_size=chunk_size,
                    min_ratio=2 / 3,
                )
                units.extend(parts)
            else:
                units.append(line)

        chunk_texts = merge_units_to_chunks(
            units=units,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for chunk_text in chunk_texts:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata=make_chunk_metadata(
                        source=document.source,
                        title=title,
                        chunk_index=chunk_index,
                    )
                )
            )

            chunk_index += 1

    return chunks
