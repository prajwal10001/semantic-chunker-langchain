# Semantic Chunker for LangChain

A **token-aware**, **LangChain-compatible** chunker that splits text (from PDF, markdown, or plain text) into semantically coherent chunks while respecting model token limits.

---

## 🚀 Features

* 🔍 **Model-Aware Token Limits**: Automatically adjusts chunking size for GPT-3.5, GPT-4, Claude, and others.
* 📄 **Multi-format Input Support**:

  * PDF via `pdfplumber`
  * Plain `.txt`
  * Markdown
  * (Extendable to `.docx` and `.html`)
* 🔁 **Overlapping Chunks**: Smart overlap between paragraphs to preserve context.
* 🧠 **Smart Merging**: Merges chunks smaller than 300 tokens.
* 🧩 **Retriever-Ready**: Direct integration with `LangChain` retrievers via FAISS.
* 🔧 **CLI Support**: Run from terminal with one command.

---

## 📆 Installation

```bash
pip install semantic-chunker-langchain
```

> Requires Python 3.9 - 3.12

---

## 🛠️ Usage

### 🔸 Chunk a PDF and Save to JSON/TXT

```bash
semantic-chunker sample.pdf --txt chunks.txt --json chunks.json
```

### 🔸 From Code

```python
from semantic_chunker_langchain.chunker import SemanticChunker, SimpleSemanticChunker
from semantic_chunker_langchain.extractors.pdf import extract_pdf
from semantic_chunker_langchain.outputs.formatter import write_to_txt

# Extract
docs = extract_pdf("sample.pdf")

# Using SemanticChunker
chunker = SemanticChunker(model_name="gpt-3.5-turbo")
chunks = chunker.split_documents(docs)

# Save to file
write_to_txt(chunks, "output.txt")

# Using SimpleSemanticChunker
simple_chunker = SimpleSemanticChunker(model_name="gpt-3.5-turbo")
simple_chunks = simple_chunker.split_documents(docs)
```

### 🔸 Convert to Retriever

```python
from langchain_community.embeddings import OpenAIEmbeddings
retriever = chunker.to_retriever(chunks, embedding=OpenAIEmbeddings())
```

---

## 📊 Testing

```bash
poetry run pytest tests/
```

---

## 👨‍💻 Authors

* Prajwal Shivaji Mandale
* Sudhnwa Ghorpade

---

## 📜 License

This project is licensed under the MIT License.
