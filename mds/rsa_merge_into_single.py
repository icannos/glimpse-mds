import argparse
import datetime
from pathlib import Path

import pandas as pd
from datasets import Dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", type=Path)

    parser.add_argument("--output_dir", type=str, default="output")

    # limit the number of samples to generate
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    path = Path(args.summaries)

    for file in path.glob("*.csv"):
        model_name, dataset, decoding_config, date, reranking_type = file.stem.split('-_-')
        df = pd.read_csv(file)
        df = df.drop(["Unnamed: 0.1",  "Unnamed: 0", ], axis=1)

        # df = df[['id', 'id_text', 'text', 'summary', 'gold']]


        merged_summaries = df.groupby("id").agg({"summary": " ".join}).reset_index()

        # add gold and text

        merged_summaries = merged_summaries.merge(df[["id", "gold", "text"]], on="id")









if __name__ == "__main__":
    main()
