from openai import OpenAI, InternalServerError
from typing import Tuple, Union
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
parser.add_argument("--num_retries", type=int, default=3)
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


def clean_response(response: str) -> Tuple[str, bool, int]:
    # Look for <think> block
    think_match = re.search(r"<think>(.*?)</think>", response, flags=re.DOTALL)
    think_present = bool(think_match)
    think_length = len(think_match.group(1).strip()) if think_present else 0

    # Only look for the answer *after* </think>
    if think_present:
        response_tail = response.split("</think>", 1)[-1]
    else:
        response_tail = response

    match = re.search(r"(yes|no|maybe)", response_tail, re.IGNORECASE | re.MULTILINE)

    if not match:
        print("Neither 'Yes', 'No', nor 'Maybe' found in the response tail:")
        print("==== Response ====")
        print(response)
        print("==================")
        return "none", think_present, think_length

    return match.group(0).lower(), think_present, think_length


if __name__ == "__main__":
    test_set = load_pubmedqa_test_set(args.test_set)
    output_file = f"{model}{args.output_prefix}-pubmedqa_results.csv"
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["key", "answer", "truth", "reasoning", "reasoning_length"])

        for key, i in test_set.items():
            for attempt in range(args.num_retries):
                try:
                    query = construct_pubmedqa_query(i)
                    response, usage = submit_query(query)
                    answer, think_present, think_length = clean_response(response)
                    truth = get_pubmedqa_answer(i)
                    success = True
                    break
                except InternalServerError:
                    continue
            if not success:
                query = None
                response = None
                usage = 0
                answer = None
                think_present = False
                think_length = 0
            
            print(
                f"Key: {key} Answer: {answer} Truth: {truth} Reasoning: {think_present} Reasoning Length: {think_length} Prompt Tokens: {usage.prompt_tokens} Completion Tokens: {usage.completion_tokens} Total Tokens: {usage.total_tokens}"
            )
            writer.writerow([key, answer, truth, think_present, think_length])

    print(f"Results saved to {output_file}")
