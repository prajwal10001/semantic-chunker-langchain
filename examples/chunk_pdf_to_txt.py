import pdfplumber
from langchain_core.documents import Document
from langchain_semantic_chunker import SemanticChunker

PDF_PATH = "ml.pdf"
OUTPUT_TXT = "output.txt"

# === 1. Extract paragraphs per page ===
def extract_text(path):
    with pdfplumber.open(path) as pdf:
        return [
            Document(page_content=page.extract_text(), metadata={"source": path, "page_number": i+1})
            for i, page in enumerate(pdf.pages) if page.extract_text()
        ]

# === 2. Chunk semantically (token-aware) ===
def chunk_documents(docs, max_tokens=1500, overlap=100):
    chunker = SemanticChunker(max_tokens=max_tokens, overlap=overlap)
    return chunker.split_documents(docs)

# === 3. Save chunks to .txt ===
def write_to_txt(chunks, path="output.txt"):
    with open(path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(chunks):
            content = doc.page_content.strip().replace("•", "-")
            content = content.replace("\n", "\n\n")
            meta = f"# Chunk {i+1} | Page: {doc.metadata.get('page_number', 'N/A')}\n"
            f.write(f"\n\n{meta}{content}\n")

# === Run ===
if __name__ == "__main__":
    docs = extract_text(PDF_PATH)
    chunks = chunk_documents(docs)
    write_to_txt(chunks, OUTPUT_TXT)
    print(f"✅ PDF content chunked and saved to {OUTPUT_TXT}")
