"""Command-line entry point for the knowledge assistant."""

from __future__ import annotations

import argparse
from gettext import install
from pathlib import Path

from answerer import Answerer
from memory import ConversationMemory
from retriever import TfidfRetriever, load_documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask questions about a local knowledge base.")
    parser.add_argument("--knowledge", default="knowledge", help="Directory containing source files")
    parser.add_argument("--memory", default="data/memory.json", help="Conversation history JSON path")
    parser.add_argument("--question", help="Ask one question and exit")
    parser.add_argument("--clear-memory", action="store_true", help="Delete saved conversation history")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    memory = ConversationMemory(args.memory)
    if args.clear_memory:
        memory.clear()
        print("Conversation memory cleared.")
        return

    documents = load_documents(Path(args.knowledge))
    if not documents:
        print(f"No supported documents found in {args.knowledge!r}.")
        print("Add .txt, .md, .csv, or .pdf files and run again.")
        return
    retriever = TfidfRetriever(documents)
    answerer = Answerer(memory)

    def respond(question: str) -> None:
        retrieval_query = question
        if memory.messages:
            retrieval_query = f"{memory.context(limit=4)}\n{question}"
        print(answerer.answer(question, retriever.search(retrieval_query)))

    if args.question:
        respond(args.question)
        return
    print(
        f"Indexed {len(documents)} document chunks with {retriever.backend}. "
        "Type 'exit' to quit."
    )
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            print("\nAssistant: ", end="")
            respond(question)
if __name__ == "__main__":
    main()
    