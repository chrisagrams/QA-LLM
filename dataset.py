import json

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