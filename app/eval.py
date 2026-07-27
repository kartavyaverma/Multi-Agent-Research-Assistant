from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from app.agents.graph import run_research

EVAL_QUESTIONS = [
    "What are the main causes of coral reef bleaching?",
    "How does intermittent fasting affect insulin sensitivity?",
    "What caused the 2008 financial crisis?",
]


def build_eval_dataset(questions: list[str]) -> Dataset:
    rows = {"question": [], "answer": [], "contexts": []}

    for q in questions:
        result = run_research(q)
        rows["question"].append(q)
        rows["answer"].append(result.get("final_answer", ""))
        rows["contexts"].append([result.get("search_results", "")])

    return Dataset.from_dict(rows)


def main():
    dataset = build_eval_dataset(EVAL_QUESTIONS)
    scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    print(scores)
    scores.to_pandas().to_csv("eval_results.csv", index=False)
    print("Saved detailed results to eval_results.csv")


if __name__ == "__main__":
    main()
