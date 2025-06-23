# === examples/retriever_demo.py ===
import os
os.environ["OPENAI_API_KEY"] = ""

from langchain_core.documents import Document
from langchain_semantic_chunker.chunker import SemanticChunker, SimpleSemanticChunker
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pdfplumber

# 1. Extract text from PDF
def extract_text_from_pdf(path):
    with pdfplumber.open(path) as pdf:
        return [
            Document(page_content=page.extract_text(), metadata={"page_number": i+1})
            for i, page in enumerate(pdf.pages) if page.extract_text()
        ]

# 2. Run the chunker
chunker = SemanticChunker(model_name="gpt-3.5-turbo")
docs = extract_text_from_pdf("sample1.pdf")
chunks = chunker.split_documents(docs)

# 3. Embed and index using FAISS
embedding_model = OpenAIEmbeddings()
retriever = chunker.to_retriever(chunks, embedding=embedding_model)

# 4. Query the retriever
query = "What is this document about?"
results = retriever.get_relevant_documents(query)

# 5. Show results
for i, doc in enumerate(results):
    print(f"\n🔍 Result {i+1} | Page {doc.metadata.get('page_number')} | Score: {doc.metadata.get('score', 'N/A')}\n{doc.page_content[:300]}...")
