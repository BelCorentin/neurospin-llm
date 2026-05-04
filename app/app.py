"""
NeuroSpin Wiki RAG — Chainlit Application

Provides a chat interface backed by the RAG pipeline in rag.py.
Each answer is streamed token-by-token, followed by expandable source citations.
"""

import chainlit as cl
from rag import query_rag, Source

WELCOME_MESSAGE = """\
## 🧠 NeuroSpin Wiki Assistant

Je suis un assistant basé sur le wiki NeuroSpin. Posez-moi vos questions en **anglais ou en français**.

I am an assistant based on the NeuroSpin wiki. Ask me anything in **English or French**.

> ℹ️ My answers are grounded in wiki pages — I will show you the source pages I used.

---
**Examples / Exemples :**
- *How do I install FSL on the cluster?*
- *Comment accéder au VPN NeuroSpin depuis chez moi ?*
- *What is the BIDS naming convention for functional data?*
- *Quels sont les quotas de stockage sur le cluster ?*
"""


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_message
async def on_message(message: cl.Message):
    question = message.content.strip()
    if not question:
        return

    # Stream the answer
    answer_msg = cl.Message(content="")
    await answer_msg.send()

    try:
        token_stream, sources = await query_rag(question)

        full_answer = ""
        async for token in token_stream:
            full_answer += token
            await answer_msg.stream_token(token)

        await answer_msg.update()

        # Render source citations as expandable elements
        if sources:
            elements = []
            for i, src in enumerate(sources):
                label = f"📄 {src.page_title} (score: {src.score:.2f})"
                content = (
                    f"**Page:** {src.page_title}\n"
                    f"**File:** {src.source_file}\n"
                    f"**Chunk:** {src.chunk_index}\n"
                    f"**Relevance score:** {src.score:.4f}\n\n"
                    f"**Excerpt:**\n\n{src.excerpt}…"
                )
                elements.append(
                    cl.Text(name=label, content=content, display="side")
                )

            await cl.Message(
                content="**Sources used:**",
                elements=elements,
            ).send()

    except Exception as e:
        error_text = str(e)
        # Surface connection errors in a user-friendly way
        if "Connection refused" in error_text or "connect" in error_text.lower():
            await cl.Message(
                content=(
                    "⚠️ Could not reach the language model service. "
                    "Please make sure vLLM is running (`docker compose up -d vllm`)."
                )
            ).send()
        elif "collection" in error_text.lower() or "not found" in error_text.lower():
            await cl.Message(
                content=(
                    "⚠️ The knowledge base is empty. "
                    "Please run the ingestion pipeline first: "
                    "`cd ingest && python ingest.py`"
                )
            ).send()
        else:
            await cl.Message(content=f"⚠️ Error: {error_text}").send()
        raise
