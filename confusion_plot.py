import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

parser = argparse.ArgumentParser()
parser.add_argument("file_name")
parser.add_argument("--title", default="Confusion Matrix")
args = parser.parse_args()

if __name__ == "__main__":
    df = pd.read_csv(args.file_name)

    class_labels = ["yes", "maybe", "no"]

    y_true = df["truth"].values
    y_pred = df["answer"].values
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(args.title)
    plt.savefig(f"{args.title}-confusion-matrix.png")