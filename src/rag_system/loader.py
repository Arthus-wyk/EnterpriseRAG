from dataclasses import dataclass
from pathlib import Path

SUPPORTED_DOCUMENT_SUFFIXES = {".txt", ".md"}


@dataclass(frozen=True)
class Document:
    source: str
    text: str


def iter_document_paths(
    directory: Path,
    suffixes: set[str] = SUPPORTED_DOCUMENT_SUFFIXES,
) -> list[Path]:
    if not directory.exists():
        raise RuntimeError(f"Document directory not found: {directory}")

    if not directory.is_dir():
        raise RuntimeError(f"Document path is not a directory: {directory}")

    return [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in suffixes
    ]


def load_documents(
    directory: Path,
    suffixes: set[str] = SUPPORTED_DOCUMENT_SUFFIXES,
    progress_every: int = 0,
) -> list[Document]:
    """Load text documents from a directory and all nested subdirectories."""
    if not directory.exists():
        raise RuntimeError(f"Document directory not found: {directory}")

    if not directory.is_dir():
        raise RuntimeError(f"Document path is not a directory: {directory}")

    documents: list[Document] = []
    scanned_count = 0

    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue

        scanned_count += 1
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            documents.append(Document(source=str(path), text=text))
        if progress_every and scanned_count % progress_every == 0:
            print(
                f"[load] scanned {scanned_count} files, loaded {len(documents)} documents",
                flush=True,
            )

    if progress_every:
        print(
            f"[load] finished scanning {scanned_count} files, loaded {len(documents)} documents",
            flush=True,
        )

    return documents
