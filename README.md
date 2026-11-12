# Token Optimization in a RAG Knowledge Assistant

This project investigates how context compression can reduce token consumption in a Retrieval-Augmented Generation (RAG) system while maintaining answer quality.

Developed as part of the technical internship exercise for imec.

## Approach

The prototype implements two RAG configurations:

- **Baseline:** Retrieves the top 5 most relevant document chunks using semantic similarity.
- **Optimized:** Uses the same retrieved chunks but applies paragraph-level context compression before generating an answer.

The implementation uses Python, Ollama, Qwen3, and Nomic Embed Text.

## Evaluation

Both configurations are evaluated using 10 questions based on laboratory operational knowledge.

The comparison includes:
- Input and output token consumption
- Response time
- Factual correctness and faithfulness
- Context precision and recall

Answer quality is evaluated using RAGAS.

## Results

The optimized approach achieved a 23.3% reduction in total token consumption and a 26.5% reduction in input tokens.

Factual correctness and faithfulness slightly improved, while context precision decreased and response time increased.

These findings highlight the trade-offs between token efficiency, answer quality, and computational overhead.

## Running the Project

The project requires Python, Ollama, and the dependencies listed in `pyproject.toml`.

Install the required Ollama models:

```bash
ollama pull qwen3:4b-instruct
ollama pull nomic-embed-text
```

Install Python dependencies using uv:

```bash
uv sync
```

Run the interactive RAG comparison:

```bash
uv run python rag.py
```

Run the evaluation:

```bash
uv run python experiment.py
```

The evaluation generates `results.csv`.

## Files

- `rag.py` — RAG pipeline and context compression
- `experiment.py` — Baseline comparison and RAGAS evaluation
- `docs/` — Sample operational knowledge documents
- `results.csv` — Experimental results
- `report.pdf` — Technical investigation and evaluation report

## Limitations

The experiment uses a small set of 10 questions and a heuristic token budget for context compression. Further experiments with larger datasets and different compression limits would help assess the reliability of the results.
