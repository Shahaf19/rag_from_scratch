"""Stage 2 of the RAG pipeline: split page Documents into smaller, overlapping chunk Documents."""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from load import load_documents

CHUNK_SIZE = 1000     # max characters per chunk: roughly a paragraph, focused enough for a sharp embedding
CHUNK_OVERLAP = 200   # ~20%: a sentence cut at a chunk boundary still appears whole in one of the two chunks


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Record where each chunk starts within its page, so we can see the overlap and trace chunks back.
        add_start_index=True,
    )
    return splitter.split_documents(docs)


if __name__ == "__main__":
    pages = load_documents()
    chunks = split_documents(pages)

    sizes = [len(c.page_content) for c in chunks]
    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    print(f"Chunk size: min {min(sizes)}, avg {sum(sizes) // len(sizes)}, max {max(sizes)}")
    print()

    # The first two chunks of page 1: the end of chunk 0 should reappear at the start of chunk 1.
    for i in range(2):
        c = chunks[i]
        print(f"=== chunk {i} | page {c.metadata['page']} | start_index {c.metadata['start_index']} "
              f"| {len(c.page_content)} chars ===")
        print(c.page_content)
        print()
