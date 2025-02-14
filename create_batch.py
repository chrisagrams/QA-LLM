import argparse
import json
from dataset import *

parser = argparse.ArgumentParser(description="Create OpenAI API batch request.")
parser.add_argument("--test_set", default="data/pubmedqa/test_set.json")
parser.add_argument("--model")

args = parser.parse_args()


def generate_query(custom_id: str , model: str, query: str, max_tokens: int) -> dict:
    query_dict = {}
    query_dict['custom_id'] = custom_id
    query_dict['method'] = "POST"
    query_dict['url'] = '/v1/chat/completions'

    body = {}
    body['model'] = model
    body["messages"] = [{"role": "user", "content": query}]
    # body["max_tokens"] = max_tokens

    query_dict['body'] = body
    return query_dict


if __name__ == "__main__":
    test_set = load_pubmedqa_test_set(args.test_set)

    with open(f"pubmedqa-{args.model}-batch.jsonl", "w") as f:
        for key, i in test_set.items():
            query = construct_pubmedqa_query(i)
            api_query = generate_query(key, args.model, query, 1000)
            f.write(json.dumps(api_query))  
            f.write("\n")
