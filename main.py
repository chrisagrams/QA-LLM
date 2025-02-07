from openai import OpenAI
import json
import re
import csv
import argparse

parser = argparse.ArgumentParser(description="Run PubMedQA on ChatGPT compatable API.")
parser.add_argument(
    "--url", help="Base URL of API.", default="http://localhost:1234/v1"
)
parser.add_argument("--api_key", default="token-abc123")
parser.add_argument("--model")
parser.add_argument("--test_set", default="data/pubmedqa/test_set.json")
parser.add_argument("--output_prefix", default="")
args = parser.parse_args()

client = OpenAI(
    base_url=args.url,
    api_key=args.api_key,
)

model = args.model


def load_pubmedqa_test_set(path: str) -> json:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def construct_pubmedqa_query(elem: dict) -> str:
    base_prompt = "For the question, the reference context, and the answer given below, is it possible to infer the answer for that question from the reference context? Only reply as either Yes or No or Maybe."
    question = elem["QUESTION"]
    context = "\n".join(elem["CONTEXTS"])

    return (
        f"{base_prompt}\nQuestion: {question}\nReference context: {context}\nAnswer: "
    )


def get_pubmedqa_answer(elem: dict) -> str:
    if "final_decision" not in elem:
        raise ValueError("Missing 'final_decision' in input dict")
    return elem["final_decision"]


def submit_query(query: str) -> str:
    completion = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": query}]
    )
    message = completion.choices[0].message.content
    return message


def clean_response(response: str) -> str:
    # Remove content inside <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

    # Find 'Yes' or 'No'
    match = re.search(r"\b(Yes|No|Maybe)\b", text, re.IGNORECASE)

    if not match:
        raise ValueError("Neither 'Yes', 'No', nor 'Maybe' found in the response.")

    return match.group(0).lower()


if __name__ == "__main__":
    test_set = load_pubmedqa_test_set(args.test_set)
    output_file = f"{model}{args.output_prefix}-pubmedqa_results.csv"
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["key", "answer", "truth"])

        for key, i in test_set.items():
            try:
                query = construct_pubmedqa_query(i)
                response = submit_query(query)
                answer = clean_response(response)
            except Exception:
                answer = None

            truth = get_pubmedqa_answer(i)
            print(f"Key: {key} Answer: {answer} Truth: {truth}")
            writer.writerow([key, answer, truth])

    print(f"Results saved to {output_file}")
