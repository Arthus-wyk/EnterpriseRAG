import re
from pathlib import Path
import matplotlib.pyplot as plt

from src.rag_system.loader import load_documents


def count_words(text: str) -> int:
    """
    统计文本字数：
    - 中文：每个汉字算 1 个字
    - 英文/数字：连续字符串算 1 个词
    """
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    english_words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text)

    return len(chinese_chars) + len(english_words)


def get_document_text(document) -> str:
    """
    从 document 中提取 text。
    兼容几种常见格式：
    1. document.text
    2. document.page_content
    3. document["text"]
    4. document["page_content"]
    5. document 本身就是字符串
    """
    if isinstance(document, str):
        return document

    if hasattr(document, "text"):
        return document.text or ""

    if hasattr(document, "page_content"):
        return document.page_content or ""

    if isinstance(document, dict):
        return (
            document.get("text")
            or document.get("page_content")
            or document.get("content")
            or ""
        )

    return ""


def get_document_name(document, index: int) -> str:
    """
    尝试获取文档名称。
    如果没有文件名，就用 document_1, document_2 这种默认名称。
    """
    if hasattr(document, "metadata"):
        metadata = document.metadata or {}

        source = (
            metadata.get("source")
            or metadata.get("file_path")
            or metadata.get("filename")
        )

        if source:
            return Path(source).name

    if isinstance(document, dict):
        metadata = document.get("metadata", {}) or {}

        source = (
            metadata.get("source")
            or metadata.get("file_path")
            or metadata.get("filename")
            or document.get("filename")
            or document.get("source")
        )

        if source:
            return Path(source).name

    return f"document_{index}"


def count_documents_words_from_loaded_docs(documents: list) -> dict:
    """
    统计 load_documents(folder_path) 返回的 documents 中每篇文档的字数。
    """
    word_counts = {}

    for index, document in enumerate(documents, start=1):
        text = get_document_text(document)
        name = get_document_name(document, index)

        word_counts[name] = count_words(text)

    return word_counts


def classify_word_counts(word_counts: dict) -> dict:
    """
    将文档按字数分组：
    - 低于 2000
    - 2000 到 4000
    - 4000 以上
    """
    categories = {
        "Below 2000": 0,
        "2000-4000": 0,
        "Above 4000": 0
    }

    for count in word_counts.values():
        if count < 2000:
            categories["Below 2000"] += 1
        elif 2000 <= count <= 4000:
            categories["2000-4000"] += 1
        else:
            categories["Above 4000"] += 1

    return categories


def plot_word_count_distribution(
    categories: dict,
    output_path: str = "word_count_distribution.png"
) -> None:
    """
    画出文档字数分布柱状图，并保存为图片。
    """
    labels = list(categories.keys())
    values = list(categories.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)

    plt.title("Document Word Count Distribution")
    plt.xlabel("Word Count Range")
    plt.ylabel("Number of Documents")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(height),
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


def analyze_document_word_distribution(folder_path: str) -> dict:
    """
    主函数：
    1. 用 load_documents(folder_path) 加载文档
    2. 统计每篇文档字数
    3. 按区间分类
    4. 输出柱状图
    """
    documents = load_documents(folder_path)

    word_counts = count_documents_words_from_loaded_docs(documents)
    categories = classify_word_counts(word_counts)

    print("每个文档的字数：")
    for filename, count in word_counts.items():
        print(f"{filename}: {count}")

    print("\n分类统计结果：")
    for category, count in categories.items():
        print(f"{category}: {count}")

    plot_word_count_distribution(categories)

    return {
        "word_counts": word_counts,
        "categories": categories
    }


if __name__ == "__main__":
    folder_path = Path(__file__).resolve().parents[2]/"all_documents"/"github"

    result = analyze_document_word_distribution(folder_path)