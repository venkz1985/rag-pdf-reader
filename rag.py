from collections.abc import AsyncGenerator

import numpy as np
from openai import AsyncOpenAI

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL, TOP_K
from ingest import get_embedding_model

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided document context.

CONTEXT FROM DOCUMENTS:
{context}

INSTRUCTIONS:
- Answer the user's question based on the context above.
- Cite which document(s) your answer comes from using [Source: filename] format.
- If the context does not contain enough information to answer, say "I don't have enough information in the uploaded documents to answer that question."
- Be concise and accurate."""

client = AsyncOpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)


def retrieve(query: str, index, metadata: list[dict], top_k: int = TOP_K) -> list[dict]:
    model = get_embedding_model()
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype="float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        chunk = metadata[idx].copy()
        chunk["score"] = float(scores[0][i])
        results.append(chunk)

    return results


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] Source: {chunk['source_file']}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


async def generate(query: str, context_chunks: list[dict], chat_history: list[dict]) -> AsyncGenerator[str, None]:
    context = build_context(context_chunks)
    system_message = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    messages = [{"role": "system", "content": system_message}]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query})

    stream = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content
