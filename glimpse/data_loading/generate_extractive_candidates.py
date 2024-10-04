import argparse
import datetime
from pathlib import Path

import pandas as pd
from datasets import Dataset
from tqdm import tqdm

import nltk
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="2017")
    parser.add_argument("--dataset_path", type=str, default="data")


    parser.add_argument("--output_dir", type=str, default="output")

    # limit the number of samples to generate
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    return args


def prepare_dataset(dataset_name, dataset_path=None) -> Dataset:
    if dataset_path is not None:
        dataset_path = Path(dataset_path)
    if dataset_name == "2017":
        dataset = pd.read_csv(dataset_path / "all_reviews_2017_sep.csv")
    elif dataset_name == "2018":
        dataset = pd.read_csv(dataset_path / "all_reviews_2018_sep.csv")
    elif dataset_name == "2019":
        dataset = pd.read_csv(dataset_path / "all_reviews_2019_sep.csv")
    elif dataset_name == "2020":
        dataset = pd.read_csv(dataset_path / "all_reviews_2020.csv")
    elif dataset_name == "2021":
        dataset = pd.read_csv(dataset_path / "all_reviews_2021.csv")
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    # make a dataset from the dataframe
    dataset = Dataset.from_pandas(dataset)

    return dataset


def evaluate_summarizer(dataset: Dataset) -> Dataset:
    """
    @param dataset: A dataset with the text
    @return: The same dataset with the summaries added
    """
    # create a dataset with the text and the summary
    # TODO: text??

    # create a dataloader

    # generate summaries
    summaries = []
    print("Generating summaries...")

    # (tqdm library for progress bar) 
    for sample in tqdm(dataset):
        text = sample["review"] # TODO: changed from 'text' to 'review', it worked

        text = re.sub(r'-{4,}', '\n', text) # replace long dashes with new line
        sentences = nltk.sent_tokenize(text)
        # remove empty sentences
        sentences = [sentence for sentence in sentences if sentence != ""]

        summaries.append(sentences)

    # add summaries to the huggingface dataset
    dataset = dataset.map(lambda example: {"summary": summaries.pop(0)})

    return dataset


def main():
    args = parse_args()
    # load the dataset
    print("Loading dataset...")
    dataset = prepare_dataset(args.dataset_name, args.dataset_path)

    # limit the number of samples
    if args.limit is not None:
        _lim = min(args.limit, len(dataset))
        dataset = dataset.select(range(_lim))

    # generate summaries
    dataset = evaluate_summarizer(
        dataset,
    )

    df_dataset = dataset.to_pandas()
    df_dataset = df_dataset.explode("summary")
    df_dataset = df_dataset.reset_index()
    # add an idx with  the id of the summary for each example
    df_dataset["id_candidate"] = df_dataset.groupby(["index"]).cumcount()

    # save the dataset
    # add unique date in name
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S")
    output_path = (
        Path(args.output_dir)
        / f"extractive_sentences-_-{args.dataset_name}-_-none-_-{date}.csv"
    )

    # create output dir if it doesn't exist
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df_dataset.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
