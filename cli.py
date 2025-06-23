import argparse
from semantic_chunker_langchain.extractors.pdf import extract_pdf
from semantic_chunker_langchain.chunker import SemanticChunker
from semantic_chunker_langchain.outputs.formatter import write_to_txt, write_to_json

def main():
    parser = argparse.ArgumentParser(description="Chunk a PDF semantically and output to .txt/.json.")
    parser.add_argument("file", help="Path to input PDF file")
    parser.add_argument("--txt", help="Path to output .txt file", default="output.txt")
    parser.add_argument("--json", help="Path to output .json file", default="output.json")
    args = parser.parse_args()

    docs = extract_pdf(args.file)
    chunker = SemanticChunker(max_tokens=1500, overlap=100)
    chunks = chunker.split_documents(docs)

    write_to_txt(chunks, args.txt)
    write_to_json(chunks, args.json)

    print(f"Done. Chunks written to {args.txt} and {args.json}")

if __name__ == "__main__":
    main()
