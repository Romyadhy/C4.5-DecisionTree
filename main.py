import datetime
import logging
import os
import numpy as np
import pandas as pd
from core.decison_tree import C45Decisiontree
from utils.data_loader import load_dataset
from utils.visualizer import export_tree_to_image
from utils.visualizer import export_tree_to_dot


FILE_PATH = "dataset/data2.csv"
TARGET_COLUMN = "PlayTennis"
LOG_DIR = "logs"
OUTPUT_DIR = "output"

# logs
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "runtime.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    print("--- C4.5 Decision Tree From Scratch ---")

    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"-- Starting Run: {run_id} --")
    logging.info(f"Started execution ID: {run_id}")

    logging.info(f"Loading dataset from {FILE_PATH}")
    X, y = load_dataset(FILE_PATH, TARGET_COLUMN)

    df_headers = pd.read_csv(FILE_PATH, nrows=0)
    feature_names = [col for col in df_headers.columns if col != TARGET_COLUMN]

    # ifdataset not found
    if X is None:
        logging.info("Dataset X not found. Exiting.")
        return
    if y is None:
        logging.info("Dataset y not found, Exiting")
        return

    # init model (hyperparametter)
    logging.info("Initializing Model with max_depth=5, min_samples_split=2")
    clf = C45Decisiontree(min_sample_split=2, max_depth=5)

    # training
    print("Training the tree")
    logging.info("Training started...")
    clf.fit(X, y)
    print("Training complated")
    logging.info("Training completed...")

    # tesing
    print("\nTesting on first 5 rows:")
    predictions = clf.predict(X[:5])
    print("Predicted:", predictions)
    print("Actual:   ", y[:5])

    all_preds = clf.predict(X)
    acc = np.sum(all_preds == y) / len(y)
    print(f"\nAccuracy on training set: {acc * 100:.2f}%")

    # to img
    print("\nGenerating tree image...")
    export_tree_to_image(
        clf.root, filename="iris_tree_c45", feature_names=feature_names
    )

    # to dot (if-else)
    output_filename = f"tree_result_{run_id}.dot"
    full_output_path = os.path.join(OUTPUT_DIR, output_filename)

    logging.info(f"Saving output to {full_output_path}...")
    print(f"\nSaving readable tree structure to: {full_output_path}")

    export_tree_to_dot(clf.root, full_output_path, feature_names)

    print("Done.")


if __name__ == "__main__":
    main()
