from langchain_core.documents import Document
import pdfplumber

def extract_pdf(path: str) -> list[Document]:
    with pdfplumber.open(path) as pdf:
        return [
            Document(page_content=page.extract_text(), metadata={"page_number": i+1})
            for i, page in enumerate(pdf.pages)
            if page.extract_text()
        ]
