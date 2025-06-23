import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.documents import Document
from langchain_semantic_chunker.chunker import SimpleSemanticChunker
from openai import AzureOpenAI
import pdfplumber

# === 1. Setup Azure OpenAI ===
client = AzureOpenAI(
    api_key="",
    api_version="",
    azure_endpoint=""
)

deployment_name = "gpt-35-turbo"

# === 2. Extract text from PDF ===
def extract_text_from_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "\n\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

raw_text = extract_text_from_pdf("testl.pdf")

# === 3. Chunk using your SemanticChunker ===
doc = Document(page_content=raw_text, metadata={"source": "testl.pdf"})
chunker = SimpleSemanticChunker(max_tokens=1500, overlap=100)
chunks = chunker.split_documents([doc])

# === 4. Ask Azure OpenAI for summary of each chunk ===
for i, chunk in enumerate(chunks):
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "Summarize this training course material for a student."},
            {"role": "user", "content": chunk.page_content}
        ]
    )
    print(f"\n🔹 Summary for Chunk {i+1}:\n{response.choices[0].message.content}\n")
