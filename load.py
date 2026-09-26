"""Stage 1 of the RAG pipeline: load the source PDF into LangChain Documents."""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = Path(__file__).parent / "2007_exercise_mindset_crumlanger_psych_sci.pdf"


def load_documents(path=PDF_PATH):
    loader = PyPDFLoader(str(path))
    return loader.load()


if __name__ == "__main__":
    docs = load_documents()

    print(f"Loaded {len(docs)} pages")
    print(f"Total characters: {sum(len(d.page_content) for d in docs)}")
    print()
    print("Page 1 metadata:", docs[0].metadata)
    print("Page 1 text (first 500 chars):")
    print("-" * 40)
    print(docs[0].page_content[:500])
