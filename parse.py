import pandas as pd
import argparse


def calculate_accuracy(csv_file):
    df = pd.read_csv(csv_file)

    correct = (df["answer"] == df["truth"]).sum()

    total = len(df)

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")


def calculate_reasoning_count(csv_file):
    df = pd.read_csv(csv_file)

    if 'reasoning' in df.columns:
        count = (df['reasoning'] == True).sum()
        total = len(df)
        percent_reasoning = count / total if total > 0 else 0
        print(f"Percent reasoning: {percent_reasoning::.2%}")
    else:
        print(f"No reasoning check in CSV.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate accuracy from CSV file.")
    parser.add_argument("csv_file", help="Path to the CSV file")
    args = parser.parse_args()

    calculate_accuracy(args.csv_file)
    calculate_reasoning_count(args.csv_file)