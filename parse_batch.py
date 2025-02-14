import json
import argparse
import csv
from dataset import load_pubmedqa_test_set

parser = argparse.ArgumentParser()
parser.add_argument("batch_output")
parser.add_argument("--test_set", default="data/pubmedqa/test_set.json")
parser.add_argument("--output_csv", default="results.csv", help="Path to output CSV file")
args = parser.parse_args()

if __name__ == "__main__":
    test_set = load_pubmedqa_test_set(args.test_set)
    
    with open(args.batch_output, "r", encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    total = 0
    correct = 0

    with open(args.output_csv, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["key", "answer", "truth"])  # CSV header
        
        for entry in data:
            custom_id = entry.get("custom_id")
            model_response = (
                entry.get("response", {})
                .get("body", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
                .lower()
                .replace(".", "")  # Remove periods
            )

            ground_truth = (
                test_set.get(custom_id, {})
                .get("final_decision", "")
                .strip()
                .lower()
            )

            if ground_truth:
                total += 1
                if model_response == ground_truth:
                    correct += 1

            writer.writerow([custom_id, model_response, ground_truth])

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"Results saved to {args.output_csv}")