from langchain_semantic_chunker.extractors.pdf import extract_pdf
from langchain_semantic_chunker.chunker import SemanticChunker
from langchain_semantic_chunker.outputs.formatter import write_to_txt, write_to_json

docs = extract_pdf("testl.pdf")
chunker = SemanticChunker(max_tokens=1500, overlap=100)
chunks = chunker.split_documents(docs)

write_to_txt(chunks, "output.txt")

write_to_json(chunks, "output.json")

print(f"\n✅ Chunking complete. {len(chunks)} chunks saved.")
