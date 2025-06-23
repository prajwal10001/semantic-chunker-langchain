import os
from langchain_core.documents import Document
from semantic_chunker_langchain import SemanticChunker
import pdfplumber

# Paths
PDF_PATH = "sample1.pdf"
OUTPUT_TXT = "output.txt"

# 1. Extract text page-wise from PDF
def extract_text(path):
    with pdfplumber.open(path) as pdf:
        return [
            Document(
                page_content=page.extract_text(),
                metadata={"source": path, "page": i}
            )
            for i, page in enumerate(pdf.pages) if page.extract_text()
        ]

# 2. Chunk semantically (token-aware)
def chunk_documents(docs, max_tokens=1500, overlap=100):
    chunker = SemanticChunker(max_tokens=max_tokens, overlap=overlap)
    return chunker.split_documents(docs)

# 3. Write output to .txt
def save_chunks_to_txt(chunks, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            f.write(f"\n\n--- Chunk {i} ---\n\n{chunk.page_content}")

# Execute
if __name__ == "__main__":
    docs = extract_text(PDF_PATH)
    chunks = chunk_documents(docs)
    save_chunks_to_txt(chunks, OUTPUT_TXT)
    print(f"✅ Chunks saved to: {OUTPUT_TXT}")
