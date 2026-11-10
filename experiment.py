
import csv
import time
import requests
from statistics import mean
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    FactualCorrectness, Faithfulness,
    ContextPrecision, ContextRecall
)
from rag import retrieve, compress_context, OLLAMA, CHAT_MODEL


# 1. TEST DATA: question + correct reference answer
tests = [
    ("How often is equipment maintained?",
     "Shared equipment receives preventive maintenance once per month."),

    ("How long do laboratory access requests take?",
     "Access requests are normally processed within three working days."),

    ("Where should confidential research data be stored?",
     "Confidential data must be stored in approved institutional storage systems, not personal cloud accounts."),

    ("Can researchers share their passwords?",
     "No. Passwords must never be shared with colleagues."),

    ("Who approves equipment returning to service?",
     "The equipment supervisor authorizes equipment to return to service."),

    ("What is required before operating specialized equipment?",
     "Researchers need general safety training, equipment-specific training, supervisor approval, an active access card, and confirmation that equipment is available and not under maintenance. Authorization must be recorded."),

    ("What should a researcher do if equipment malfunctions during an experiment?",
     "Stop safely, secure equipment, record errors, preserve logs, notify both supervisors, submit a maintenance ticket, and document the incident. Restart only after authorization. Before resuming, verify availability, calibration and safety."),

    ("Can reserved equipment be used during maintenance?",
     "No. Maintenance restrictions take priority over equipment reservations."),

    ("How long must experiment records be retained?",
     "At least five years after project completion."),

    ("What is the annual salary of a laboratory supervisor?",
     "I don't know.")
]


# 2. SET UP RAGAS EVALUATOR
client = AsyncOpenAI(
    api_key="ollama",
    base_url=f"{OLLAMA}/v1",
    timeout=180.0
)

judge = llm_factory(CHAT_MODEL, provider="openai", client=client)

metrics = {
    "correctness": FactualCorrectness(llm=judge, mode="f1"),
    "faithfulness": Faithfulness(llm=judge),
    "context_precision": ContextPrecision(llm=judge),
    "context_recall": ContextRecall(llm=judge)
}


# 3. GENERATE ANSWER
def ask(prompt):
    start = time.perf_counter()

    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=180
    )
    r.raise_for_status()
    data = r.json()

    return (
        data["message"]["content"],
        data["prompt_eval_count"],
        data["eval_count"],
        time.perf_counter() - start
    )


# 4. RUN RAGAS EVALUATION
def evaluate(question, answer, reference, contexts):
    scores = {}

    for name, metric in metrics.items():
        arguments = {
            "user_input": question,
            "response": answer,
            "reference": reference,
            "retrieved_contexts": contexts
        }

        # Each metric only receives the fields it needs
        if name == "correctness":
            arguments = {
                "response": answer,
                "reference": reference
            }
        elif name == "faithfulness":
            arguments.pop("reference")
        else:
            arguments.pop("response")

        try:
            scores[name] = round(float(metric.score(**arguments).value), 3)
        except Exception as e:
            print(f"{name} failed: {e}")
            scores[name] = None

    return scores


# 5. RUN EXPERIMENT
results = []

for question, reference in tests:
    print("\nQuestion:", question)

    start = time.perf_counter()
    retrieved = retrieve(question, k=5)
    retrieval_time = time.perf_counter() - start

    baseline = [f"[{source}] {chunk}" for chunk, source in retrieved]

    start = time.perf_counter()
    compressed = compress_context(question, retrieved, max_tokens=200)
    compression_time = time.perf_counter() - start

    optimized = compressed.split("\n\n") if compressed else []

    for method, contexts in [
        ("Baseline", baseline),
        ("Optimized", optimized)
    ]:
        prompt = f"""Answer using ONLY the context below.
If the answer is missing, say "I don't know".

Context:
{chr(10).join(contexts)}

Question: {question}"""

        answer, input_tokens, output_tokens, llm_time = ask(prompt)

        # For the salary question, check abstention separately
        if "annual salary" in question:
            scores = {name: None for name in metrics}
            abstained = "i don't know" in answer.lower()
        else:
            scores = evaluate(question, answer, reference, contexts)
            abstained = None

        total_time = retrieval_time + llm_time
        if method == "Optimized":
            total_time += compression_time

        results.append({
            "question": question,
            "method": method,
            "reference": reference,
            "answer": answer,
            "contexts": "\n\n".join(contexts),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "time_seconds": round(total_time, 2),
            **scores,
            "correct_abstention": abstained
        })

        print(method, "| Tokens:", input_tokens + output_tokens,
              "| RAGAS:", scores)


# 6. SAVE RESULTS
with open("results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)


# 7. SUMMARY
print("\n--- EXPERIMENT SUMMARY ---")

def average(method, field):
    values = [
        float(r[field]) for r in results
        if r["method"] == method and r[field] is not None
    ]
    return mean(values) if values else None


for field in [
    "input_tokens", "output_tokens", "total_tokens",
    "time_seconds", *metrics
]:
    baseline_avg = average("Baseline", field)
    optimized_avg = average("Optimized", field)

    if baseline_avg is not None and optimized_avg is not None:
        print(f"{field}: {baseline_avg:.3f} -> {optimized_avg:.3f}")

baseline_total = average("Baseline", "total_tokens")
optimized_total = average("Optimized", "total_tokens")

reduction = (baseline_total - optimized_total) / baseline_total * 100

print(f"\nGeneration token reduction: {reduction:.1f}%")
print("Results saved to results.csv")
