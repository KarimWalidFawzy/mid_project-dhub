from pathlib import Path

from answerer import Answerer
from memory import ConversationMemory
from retriever import TfidfRetriever, load_documents


def test_retrieval_and_answer(tmp_path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "policy.txt").write_text(
        "Refund policy: all purchases are refundable within 30 days. "
        "Customer support can help with returns.",
        encoding="utf-8",
    )

    documents = load_documents(knowledge_dir)
    retriever = TfidfRetriever(documents)
    results = retriever.search("refund policy")

    assert results
    assert "refund" in results[0].document.text.lower()

    memory = ConversationMemory(tmp_path / "memory.json")
    answerer = Answerer(memory)
    answer = answerer.answer("What is the refund policy?", results)

    assert "refund" in answer.lower()
    assert "Sources:" in answer
