from semantic_chunker_langchain.extractors.pdf import extract_pdf
from semantic_chunker_langchain.chunker import SemanticChunker
from semantic_chunker_langchain.outputs.formatter import write_to_txt, write_to_json

docs = extract_pdf("testl.pdf")
chunker = SemanticChunker(max_tokens=1500, overlap=100)
chunks = chunker.split_documents(docs)

write_to_txt(chunks, "output.txt")

write_to_json(chunks, "output.json")

print(f"\n Chunking complete. {len(chunks)} chunks saved.")
