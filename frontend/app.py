import os

import streamlit as st
from dotenv import load_dotenv

from api_client import ApiClient, ApiClientError

load_dotenv()

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="centered",
)

client = ApiClient(os.getenv("API_BASE_URL"))

st.title("📚 RAG Document Assistant")
st.caption("Ask questions about the course PDFs. Answers are grounded in the uploaded documents.")

with st.sidebar:
    st.header("RAG Assistant")
    st.subheader("Backend status")
    try:
        health = client.health()
        st.success("Backend connected" if health.get("status") == "ok" else "Backend unavailable")
    except ApiClientError as error:
        st.error(f"Backend unavailable: {error}")

    st.subheader("Example questions")
    examples = [
        "What is the purpose of an activation function in a neural network?",
        "What program converts assembly language to machine language?",
        "What is pattern recognition?",
    ]
    for example in examples:
        st.caption(f"• {example}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(f"- {source}")

question = st.chat_input("Ask something about your course documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the course documents..."):
            try:
                result = client.query(question)
                answer = result["answer"]
                sources = result.get("sources", [])
            except ApiClientError:
                answer = (
                    "Sorry, I couldn't reach the backend. "
                    "Please make sure the API is running and try again."
                )
                sources = []
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.write(f"- {source}")
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
