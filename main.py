from openai import OpenAI
import json
import re
import csv


client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="token-abc123",
)

model = 'deepseek-r1-distill-qwen-7b'

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
    final_decision = elem['final_decision']
    if final_decision == 'yes':
        return True
    elif final_decision == 'no':
        return False
    else:
        raise ValueError("Invalid value for 'final_decision'.")
    

def submit_query(query: str) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": query
            }
        ]
    )
    message = completion.choices[0].message.content
    return message


def clean_response(response: str) -> bool:
    # Remove content inside <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

    # Find 'Yes' or 'No'
    match = re.search(r"\b(Yes|No)\b", text, re.IGNORECASE)
    
    if not match:
        raise ValueError("Neither 'Yes' nor 'No' found in the text.")
    
    return match.group(0).lower() == "yes"


import csv

if __name__ == "__main__":
    test_set = load_pubmedqa_test_set("data/pubmedqa/test_set.json")
    output_file = "deepseek-r1-distill-qwen-7b-pubmedqa_results.csv"
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
