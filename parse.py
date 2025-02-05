import pandas as pd
import argparse

def calculate_accuracy(csv_file: str) -> float:
    df = pd.read_csv(csv_file, dtype={"answer": str, "truth": str})

    df = df.dropna(subset=["answer", "truth"])

    df["answer"] = df["answer"].str.strip().map({"True": True, "False": False})
    df["truth"] = df["truth"].str.strip().map({"True": True, "False": False})

    accuracy = (df["answer"] == df["truth"]).mean()

    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate accuracy from CSV")
    parser.add_argument("csv_file", help="Path to input CSV file")
    args = parser.parse_args()

    accuracy = calculate_accuracy(args.csv_file)

    print(f"Accuracy: {accuracy:.2%}")