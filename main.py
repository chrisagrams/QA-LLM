from openai import OpenAI
from typing import Tuple
import re
import csv
import argparse
from dataset import *

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


def submit_query(query: str) -> Tuple[str, dict]:
    completion = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": query}]
    )
    message = completion.choices[0].message.content
    usage = completion.usage
    return message, usage


def clean_response(response: str) -> str:
    # Remove content inside <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

    # Find 'Yes' or 'No'
    match = re.search(r"\b(Yes|No|Maybe)\b", text, re.IGNORECASE)

    if not match:
        print("Neither 'Yes', 'No', nor 'Maybe' found in the response.")
        return "none"

    return match.group(0).lower()


if __name__ == "__main__":
    test_set = load_pubmedqa_test_set(args.test_set)
    output_file = f"{model}{args.output_prefix}-pubmedqa_results.csv"
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["key", "answer", "truth"])

        for key, i in test_set.items():
            # try:
            query = construct_pubmedqa_query(i)
            response, usage = submit_query(query)
            answer = clean_response(response)
            # except Exception:
            #     answer = None
            #     usage = None

            truth = get_pubmedqa_answer(i)
            print(
                f"Key: {key} Answer: {answer} Truth: {truth} Prompt Tokens: {usage.prompt_tokens} Completion Tokens: {usage.completion_tokens} Total Tokens: {usage.total_tokens}"
            )
            writer.writerow([key, answer, truth])

    print(f"Results saved to {output_file}")
