import json

def write_to_txt(docs, path="output.txt"):
    with open(path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(docs):
            chunk_header = f"# Chunk {i+1}  |  Source: {doc.metadata.get('source', 'N/A')} | Page: {doc.metadata.get('page_number', 'N/A')}\n"
            
            # Format content for markdown: spacing and bullets
            content = doc.page_content.strip()
            content = content.replace("•", "-")
            content = content.replace("\n", "\n\n")  # Add blank line between lines

            # Write formatted chunk
            f.write(f"\n\n{chunk_header}{content}\n")


def write_to_json(docs, path="output.json"):
    formatted = [
        {"chunk": i+1, "content": doc.page_content.strip(), "metadata": doc.metadata}
        for i, doc in enumerate(docs)
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)
