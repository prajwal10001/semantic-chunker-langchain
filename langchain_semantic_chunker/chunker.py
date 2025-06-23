# === langchain_semantic_chunker/chunker.py ===
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
from langchain_semantic_chunker.utils import estimate_token_count
import re

class SemanticChunker(TextSplitter):
    def __init__(self, max_tokens: int = None, overlap: int = 200, model_name: str = "gpt-3.5-turbo", chunking_type: str = "text"):
        self.model_name = model_name
        self.max_tokens = max_tokens or self._default_tokens_for_model(model_name)
        self.overlap = overlap
        self.chunking_type = chunking_type

    def _default_tokens_for_model(self, model_name: str) -> int:
        if "claude" in model_name:
            return 8000
        elif "gpt-4" in model_name:
            return 4000
        else:
            return 1500

    def score_chunk(self, text: str) -> float:
        return estimate_token_count(text, model_name=self.model_name)

    def _split_paragraphs(self, text: str) -> list[str]:
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    def _split_markdown(self, text: str) -> list[str]:
        # Split at headings (e.g., ## or ###)
        return re.split(r"(?=^#{1,6}\s)", text, flags=re.MULTILINE)

    def _split_code(self, text: str) -> list[str]:
        return text.split("\n\n")  # Simple fallback chunker

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks = []

        for doc in documents:
            text = doc.page_content
            metadata = doc.metadata.copy()

            if self.chunking_type == "markdown":
                blocks = self._split_markdown(text)
            elif self.chunking_type == "code":
                blocks = self._split_code(text)
            else:
                blocks = self._split_paragraphs(text)

            current_chunk = []
            token_count = 0

            for block in blocks:
                block_tokens = estimate_token_count(block, model_name=self.model_name)

                if token_count + block_tokens > self.max_tokens:
                    chunk_text = "\n\n".join(current_chunk)
                    chunk_metadata = metadata.copy()
                    chunk_metadata["score"] = self.score_chunk(chunk_text)
                    chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))

                    if self.overlap and len(current_chunk) > 0:
                        overlap_text = current_chunk[-1]
                        overlap_tokens = estimate_token_count(overlap_text, model_name=self.model_name)
                        current_chunk = [overlap_text]
                        token_count = overlap_tokens
                    else:
                        current_chunk = []
                        token_count = 0

                current_chunk.append(block)
                token_count += block_tokens

            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunk_metadata = metadata.copy()
                chunk_metadata["score"] = self.score_chunk(chunk_text)
                chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        merged_chunks = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            token_count = estimate_token_count(chunk.page_content, model_name=self.model_name)
            if token_count < 300 and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                merged_text = chunk.page_content + "\n\n" + next_chunk.page_content
                merged_meta = chunk.metadata.copy()
                merged_meta.update(next_chunk.metadata)
                merged_meta["score"] = self.score_chunk(merged_text)
                merged_chunks.append(Document(page_content=merged_text, metadata=merged_meta))
                i += 2
            else:
                merged_chunks.append(chunk)
                i += 1

        return merged_chunks

    def split_text(self, text: str) -> list[str]:
        return self._split_paragraphs(text)

    def to_retriever(self, chunks: list[Document], embedding) -> object:
        from langchain_community.vectorstores import FAISS
        return FAISS.from_documents(chunks, embedding=embedding).as_retriever()

    
class SimpleSemanticChunker(SemanticChunker):
    def split_text(self, text):
        return text.split('\n\n')




# from langchain_core.documents import Document
# from langchain_text_splitters import TextSplitter
# from langchain_semantic_chunker.utils import estimate_token_count


# class SemanticChunker(TextSplitter):
#     def __init__(self, max_tokens: int = 1500, overlap: int = 200, model_name: str = "gpt-3.5-turbo"):
#         """
#         Token-aware document chunker for LangChain.

#         Args:
#             max_tokens (int): Maximum tokens per chunk
#             overlap (int): Optional overlap in tokens between chunks
#             model_name (str): The model name for token estimation (used with tiktoken)
#         """
#         self.max_tokens = max_tokens
#         self.overlap = overlap
#         self.model_name = model_name

#     def split_documents(self, documents: list[Document]) -> list[Document]:
#         chunks = []

#         for doc in documents:
#             text = doc.page_content
#             metadata = doc.metadata.copy()

#             paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
#             current_chunk = []
#             token_count = 0

#             for para in paragraphs:
#                 para_tokens = estimate_token_count(para, model_name=self.model_name)

#                 if token_count + para_tokens > self.max_tokens:
#                     # Commit current chunk
#                     chunk_text = "\n\n".join(current_chunk)
#                     chunks.append(Document(page_content=chunk_text, metadata=metadata))

#                     # Start new chunk with overlap (if defined)
#                     if self.overlap and len(current_chunk) > 0:
#                         overlap_text = current_chunk[-1]
#                         overlap_tokens = estimate_token_count(overlap_text, model_name=self.model_name)
#                         current_chunk = [overlap_text]
#                         token_count = overlap_tokens
#                     else:
#                         current_chunk = []
#                         token_count = 0

#                 current_chunk.append(para)
#                 token_count += para_tokens

#             if current_chunk:
#                 chunk_text = "\n\n".join(current_chunk)
#                 chunks.append(Document(page_content=chunk_text, metadata=metadata))

#         return chunks

#     def split_text(self, text: str) -> list[str]:
#         """
#         Dummy method to satisfy LangChain's abstract base class requirement.
#         """
#         return text.split('\n\n')



# class SimpleSemanticChunker(SemanticChunker):
#     def split_text(self, text):
#         # Dummy implementation: split by paragraphs
#         return text.split('\n\n')
