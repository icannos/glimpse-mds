import argparse
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
import datetime
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

GENERATION_CONFIGS = {
    "top_p_sampling": {
        "max_new_tokens": 200,
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 1.0,
        "num_return_sequences": 8,
        "num_beams" : 1,

        #"num_beam_groups" : 4,
    },

    **{
        f"sampling_topp_{str(topp).replace('.', '')}": {
            "max_new_tokens": 200,
            "do_sample": True,
            "num_return_sequences": 8,
            "top_p": 0.95,
        }
        for topp in [0.5, 0.8, 0.95, 0.99]
    },
}

# add base.csv config to all configs
for key, value in GENERATION_CONFIGS.items():
    GENERATION_CONFIGS[key] = {
        # "max_length": 2048,
        "min_length": 0,
        "early_stopping": True,
        **value,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--dataset_name", type=str, default="amazon")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--decoding_config", type=str, default="top_p_sampling", choices=GENERATION_CONFIGS.keys())

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--output_dir", type=str, default="output")

    # limit the number of samples to generate
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    return args


def prepare_dataset(dataset_name, dataset_path=None) -> Dataset:
    if dataset_path is not None:
        dataset_path = Path(dataset_path)   
    if dataset_name in range (2017,2021):
        dataset = pd.read_csv(dataset_path / f"all_reviews_{dataset_name}.csv")
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    # make a dataset from the dataframe
    dataset = Dataset.from_pandas(dataset)

    return dataset


def evaluate_summarizer(
    model, tokenizer, dataset: Dataset, decoding_config, batch_size: int
) -> Dataset:
    """
    @param model: The model used to generate the summaries
    @param tokenizer: The tokenizer used to tokenize the text and the summary
    @param dataset: A dataset with the text
    @param decoding_config: Dictoionary with the decoding config
    @param batch_size: The batch size used to generate the summaries
    @return: The same dataset with the summaries added
    """
    # create a dataset with the text and the summary

    # create a dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # generate summaries
    summaries = []
    print("Generating summaries...")

    for batch in tqdm(dataloader):
        text = batch["text"]


        inputs = tokenizer(
            text,
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # move inputs to device
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

        # generate summaries
        outputs = model.generate(
            **inputs,
            **decoding_config,
        )
        # output : (batch_size * num_return_sequences, max_length)
        outputs = outputs.reshape(batch_size, -1, outputs.shape[-1])

        # decode summaries
        for b in range(batch_size):
            summaries.append(
                [
                    tokenizer.decode(
                        outputs[b, i],
                        skip_special_tokens=True,
                    )
                    for i in range(outputs.shape[1])
                ]
            )

    # add summaries to the huggingface dataset
    dataset = dataset.map(lambda example: {"summary": summaries.pop(0)})

    return dataset


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize the model name to be used as a folder name.
    @param model_name: The model name
    @return: The sanitized model name
    """
    return model_name.replace("/", "_")


def main():
    args = parse_args()

    # load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    # move model to device
    model = model.to(args.device)

    # load the dataset
    print("Loading dataset...")
    dataset = prepare_dataset(args.dataset_name, args.dataset_path)

    # limit the number of samples
    if args.limit is not None:
        _lim = min(args.limit, len(dataset))
        dataset = dataset.select(range(_lim))

    # generate summaries
    dataset = evaluate_summarizer(
        model,
        tokenizer,
        dataset,
        GENERATION_CONFIGS[args.decoding_config],
        args.batch_size,
    )

    df_dataset = dataset.to_pandas()
    df_dataset = df_dataset.explode('summary')
    df_dataset = df_dataset.reset_index()
    # add an idx with  the id of the summary for each example
    df_dataset['id_candidate'] = df_dataset.groupby(['index']).cumcount()

    # save the dataset
    # add unique date in name
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = sanitize_model_name(args.model_name)
    output_path = (
        Path(args.output_dir)
        / f"{model_name}-_-{args.dataset_name}-_-{args.decoding_config}-_-{date}.csv"
    )

    # create output dir if it doesn't exist
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df_dataset.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
