from langchain_core.documents import Document
from langchain_semantic_chunker import SemanticChunker

def test_chunking_basic():
    doc = Document(page_content="A.\n\nB.\n\nC.", metadata={})
    chunker = SemanticChunker(max_tokens=10)
    chunks = chunker.split_documents([doc])
    assert len(chunks) >= 1
    assert all(isinstance(chunk.page_content, str) for chunk in chunks)
