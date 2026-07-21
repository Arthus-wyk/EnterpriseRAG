# Simple RAG

这是一个最小可扩展的 RAG（Retrieval-Augmented Generation）项目骨架，适合先跑通“文档加载 -> 切分 -> 建索引 -> 检索 -> 生成回答”的流程。

## 目录结构

```text
simple_rag/
  data/
    raw/          # 放原始文档，例如 .txt、.md
    index/        # 保存向量索引和元数据
  src/
    rag_system/
      config.py   # 路径与参数配置
      loader.py   # 文档读取
      splitter.py # 文本切分
      indexer.py  # 构建和保存索引
      retriever.py# 检索相关逻辑
      generator.py# 生成回答
      app.py      # 命令行入口
  tests/
  .env.example
  requirements.txt
```

## 快速开始

1. 进入目录并安装依赖

```bash
cd simple_rag
pip install -r requirements.txt
```

2. 放入文档

把 `.txt` 或 `.md` 文件放到：

```text
data/raw/
```

3. 构建索引

```bash
python -m src.rag_system.indexer
```

4. 提问

```bash
python -m src.rag_system.app "你的问题"
```

## LLM 配置

默认生成器会返回检索到的上下文摘要，不强制依赖外部 API。

如果后续要接入 OpenAI、Ollama 或企业内部模型，可以在 `src/rag_system/generator.py` 中替换 `Generator.generate()` 的实现。
