import argparse
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm

import sys, os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize the model name to be used as a folder name.
    @param model_name: The model name
    @return: The sanitized model name
    """
    return model_name.replace("/", "_")


# logging.basicConfig(stream=stdout, level=logging.)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", type=Path, default="data/GLIMPSE_results_2017.csv")

    args = parser.parse_args()
    return args



def parse_summaries(path: Path):
    """
    :return: a pandas dataframe with at least the columns 'text' and 'summary'
    """
    # read csv file

    df = pd.read_csv(path).dropna()

    # check if the csv file has the correct columns
    if not all([col in df.columns for col in ["gold", "summary"]]):
        raise ValueError("The csv file must have the columns 'text' and 'summary'.")

    return df


def evaluate_rouge(df):
    texts = df.gold.tolist()
    summaries = df.summary.tolist()

    metrics = {"rouge1": [], "rouge2": [], "rougeL": [], "rougeLsum": []}

    rouges = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )

    for summary, text in tqdm(zip(summaries, texts), total=len(texts), desc="Evaluating ROUGE"):
        scores = rouges.score(summary, text)
        metrics["rouge1"].append(scores["rouge1"].fmeasure)
        metrics["rouge2"].append(scores["rouge2"].fmeasure)
        metrics["rougeL"].append(scores["rougeL"].fmeasure)
        metrics["rougeLsum"].append(scores["rougeLsum"].fmeasure)

    return metrics


def main():
    args = parse_args()

    # load the model
    df = parse_summaries(args.summaries)

    metrics = evaluate_rouge(df)


    # # add index to the metrics
    # metrics["index"] = [i for i in range(len(df))]

    df = pd.DataFrame.from_dict(metrics)
    df = df.add_prefix(f"common/")

    # merge the metrics with the summaries

    if args.summaries.exists():
        df_old = parse_summaries(args.summaries)

        for col in df.columns:
            if col not in df_old.columns:
                df_old[col] = float("nan")

        # add entry to the dataframe
        for col in df.columns:
            df_old[col] = df[col]

        df = df_old

    df.to_csv(args.summaries, index=False)


if __name__ == "__main__":
    main()
