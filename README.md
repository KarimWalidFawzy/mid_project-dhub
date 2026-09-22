# Knowledge Assistant with Memory

A local retrieval-augmented knowledge assistant. It indexes `.txt`, `.md`,
`.csv`, and `.pdf` files, retrieves relevant passages with the
`all-MiniLM-L6-v2` sentence-transformers model, and generates grounded answers
with `google/flan-t5-small` from transformers. If either model cannot be loaded
(for example, when running offline), the assistant falls back to local
TF-IDF retrieval and extractive answers.

## Run locally

```text
python -m pip install -r requirements.txt
```

Put source files in `knowledge/`, then ask one question:

```text
python main.py --question "What is the refund policy?"
```

Run the interactive assistant with `python main.py`. Conversation history is
stored in `data/memory.json`; clear it with `python main.py --clear-memory`.

PDF ingestion requires the optional `pypdf` dependency listed above. The core
text, Markdown, and CSV workflow uses only the Python standard library.

## Docker

```text
docker compose run --rm assistant
```

Mount local files under `knowledge/`; the `data/` volume preserves memory.
