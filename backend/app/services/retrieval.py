from typing import Any


def retrieve(
    question: str,
    embedding_model: Any,
    collection: Any,
    rag_config: dict,
) -> list[dict]:
    """Retrieve chunks using the notebook's cosine-distance guard."""
    embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    )
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    k = int(rag_config["k"])
    threshold = float(rag_config["relevance_distance_threshold"])
    result = collection.query(
        query_embeddings=embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    retrieved = []
    for text, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        if distance > threshold:
            continue
        page = int(metadata["page"])
        page_end = int(metadata.get("page_end", page))
        retrieved.append(
            {
                "text": text,
                "metadata": metadata,
                "distance": distance,
                "source_label": (
                    f"{metadata['source']}, pages {page}-{page_end}"
                    if page_end != page
                    else f"{metadata['source']}, page {page}"
                ),
            }
        )
    return retrieved
