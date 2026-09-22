"""Answer generation for retrieved knowledge and conversation context."""

from __future__ import annotations
import re
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:  # pragma: no cover - exercised only without requirements installed.
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

from memory import ConversationMemory
from retriever import SearchResult


class Answerer:
    def __init__(
        self,
        memory: ConversationMemory,
        model_name: str = "google/flan-t5-small",
        use_generation: bool = True,
    ) -> None:
        self.memory = memory
        self.model_name = model_name
        self.use_generation = use_generation
        self.tokenizer = None
        self.model = None
        self._model_load_attempted = False

    def _load_model(self) -> None:
        if self._model_load_attempted:
            return
        self._model_load_attempted = True
        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        except Exception:
            self.tokenizer = None
            self.model = None

    def answer(self, question: str, results: list[SearchResult]) -> str:
        if not results:
            response = "I could not find relevant information in the knowledge base."
            self.memory.add("user", question)
            self.memory.add("assistant", response)
            return response

        question_terms = set(re.findall(r"[a-zA-Z0-9']+", question.lower()))
        passages: list[str] = []
        for result in results:
            sentences = re.split(r"(?<=[.!?])\s+", result.document.text)
            ranked = sorted(
                sentences,
                key=lambda sentence: len(
                    question_terms & set(re.findall(r"[a-zA-Z0-9']+", sentence.lower()))
                ),
                reverse=True,
            )
            passages.extend(sentence.strip() for sentence in ranked[:2] if sentence.strip())

        answer = " ".join(dict.fromkeys(passages[:4]))
        sources = ", ".join(sorted({result.document.source for result in results}))

        if self.use_generation:
            self._load_model()

        if self.tokenizer is not None and self.model is not None:
            prompt = (
                "Answer the question using only the context. "
                f"Question: {question}\nContext: {answer}"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            generated = self.model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
            )
            generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            answer = generated_text.strip() or answer

        self.memory.add("user", question)
        response = f"{answer}\n\nSources: {sources}"
        self.memory.add("assistant", response)
        return response
