from langchain_core.documents import Document
from semantic_chunker_langchain import SemanticChunker

doc = Document(
    page_content="This is the first paragraph.\n\nThis is the second, longer paragraph with more content and detail.\n\nA third paragraph for testing.",
    metadata={"source": "example"}
)

chunker = SemanticChunker(max_tokens=40, overlap=10)
chunks = chunker.split_documents([doc])

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---\n{chunk.page_content}")
