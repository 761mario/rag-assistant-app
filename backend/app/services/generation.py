from typing import Any

import ollama


PROMPT_TEMPLATE = """Use the context below to answer the question. If it contains partial information, answer with what it supports and cite it. Only say "I don't know based on the documents" if the context is unrelated. Never use outside knowledge.

Example:
Context: Source filename: lecture.pdf | page 2
An assembler converts assembly language into machine language.
Question: What does an assembler do?
Answer: An assembler converts assembly language into machine language [lecture.pdf, page 2].

Context:
{context}

Question: {question}
Answer:"""


def create_ollama_client(base_url: str) -> ollama.Client:
    return ollama.Client(host=base_url, timeout=120)


def format_context(retrieved: list[dict]) -> str:
    return "\n\n".join(
        (
            f"Source filename: {item['metadata']['source']} | "
            f"pages {item['metadata']['page']}-"
            f"{item['metadata'].get('page_end', item['metadata']['page'])}\n"
            f"{item['text']}"
        )
        for item in retrieved
    )


def generate_answer(
    question: str,
    retrieved: list[dict],
    ollama_client: ollama.Client,
    model_name: str,
) -> str:
    prompt = PROMPT_TEMPLATE.format(
        context=format_context(retrieved),
        question=question,
    )
    response = ollama_client.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return response["message"]["content"].strip()
