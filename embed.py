"""Stage 3 of the RAG pipeline: turn text into vectors and compare them by meaning."""
import math
import sys

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from load import load_documents
from split import split_documents

EMBEDDING_MODEL = "models/gemini-embedding-001"


def get_embeddings():
    # The API key is read from the GOOGLE_API_KEY environment variable, so it never lives in code.
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def cosine_similarity(a, b):
    # Only the angle between the vectors matters, not their lengths:
    # "same direction" is what the embedding model uses to mean "same meaning".
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "What happened to the blood pressure of the informed group?"

    chunks = split_documents(load_documents())
    embeddings = get_embeddings()

    # Documents and the question go through different methods: Gemini is told whether it's embedding
    # a passage (RETRIEVAL_DOCUMENT) or a query (RETRIEVAL_QUERY), so questions land near their answers.
    chunk_vectors = embeddings.embed_documents([c.page_content for c in chunks])
    question_vector = embeddings.embed_query(question)

    print(f"Embedded {len(chunk_vectors)} chunks + 1 question")
    print(f"Each vector has {len(question_vector)} numbers")
    print(f"Question vector, first 5 numbers: {[round(x, 4) for x in question_vector[:5]]}")
    print(f"Question vector length (norm): {math.sqrt(sum(x * x for x in question_vector)):.4f}")
    print()

    scored = sorted(
        ((cosine_similarity(question_vector, v), c) for v, c in zip(chunk_vectors, chunks)),
        key=lambda pair: pair[0],
        reverse=True,
    )

    print(f"Question: {question}")
    print()
    print("Most similar chunks:")
    for score, c in scored[:3]:
        print(f"  {score:.3f} | page {c.metadata['page']} | {c.page_content[:150].replace(chr(10), ' ')}...")
    print()
    print("Least similar chunks:")
    for score, c in scored[-3:]:
        print(f"  {score:.3f} | page {c.metadata['page']} | {c.page_content[:150].replace(chr(10), ' ')}...")
