"""Stage 4 of the RAG pipeline: store chunk vectors (plus their text and metadata) in Chroma."""
import sys
from pathlib import Path

from langchain_chroma import Chroma

from embed import get_embeddings
from load import load_documents
from split import split_documents

PERSIST_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "mindset_paper"


def get_vectorstore():
    embeddings = get_embeddings()

    # Reuse the saved store so we only pay for embedding once. Chroma doesn't know if the chunks
    # would come out differently now, so after changing chunk size/overlap, delete chroma_db/ to rebuild.
    if PERSIST_DIR.exists():
        print(f"Loading existing store from {PERSIST_DIR.name}/")
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(PERSIST_DIR),
        )

    print(f"Building new store in {PERSIST_DIR.name}/ (embedding all chunks)")
    chunks = split_documents(load_documents())
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        # Cosine distance = 1 - cosine similarity, so scores line up with our hand-computed ones in embed.py.
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "What happened to the blood pressure of the informed group?"

    vectorstore = get_vectorstore()
    print(f"Store holds {len(vectorstore.get()['ids'])} chunks")
    print()

    # Chroma embeds the question (via embed_query) and finds the nearest chunk vectors for us:
    # this replaces the sorted(...) loop in embed.py.
    results = vectorstore.similarity_search_with_score(question, k=3)

    print(f"Question: {question}")
    print()
    for doc, distance in results:
        print(f"  distance {distance:.3f} (similarity {1 - distance:.3f}) | page {doc.metadata['page']} "
              f"| {doc.page_content[:120].replace(chr(10), ' ')}...")
