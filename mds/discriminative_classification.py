from typing import Tuple

import numpy as np
import pandas as pd
import argparse
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

def xlogx(x):
    if x == 0:
        return 0
    else:
        return x * torch.log(x)

def parse_summaries(path : Path):

    # Load the data
    df = pd.read_csv(path)

    if 'id' not in df.columns:
        raise ValueError('id column not found in the summaries file')
    if 'text' not in df.columns:
        raise ValueError('text column not found in the summaries file')
    if 'summary' not in df.columns:
        raise ValueError('summary column not found in the summaries file')

    return df


def embed_text_and_summaries(df : pd.DataFrame, model : SentenceTransformer) -> Tuple[torch.Tensor, torch.Tensor]:

    text_embeddings = model.encode(df.text.tolist(), convert_to_tensor=True)
    summary_embeddings = model.encode(df.summary.tolist(), convert_to_tensor=True)

    return text_embeddings, summary_embeddings


def compute_dot_products(df : pd.DataFrame, text_embeddings : torch.Tensor, summary_embeddings : torch.Tensor):

    df = df.reset_index()
    df['index'] = df.index

    # group by id
    grouped = df.groupby('id')

    # for each id gather the id of the text and the summary
    ids_per_sample = grouped.index.apply(list).tolist()

    # compute the dot product between the text and the summary

    metrics = {'proba_of_success' : []}
    for text_ids in ids_per_sample:
        # shape (num_text, embedding_dim)
        text_embedding = text_embeddings[text_ids]
        summary_embedding = summary_embeddings[text_ids]

        # shape (num_text, num_text=num_summary)
        dot_product = torch.matmul(text_embedding, summary_embedding.T)

        # apply log softmax
        log_softmax = torch.nn.functional.log_softmax(dot_product, dim=0)

        # num_text
        log_proba_of_success = torch.diag(log_softmax).squeeze()
        entropy = torch.xlogy(log_proba_of_success, log_proba_of_success).sum(0).squeeze()

        metrics['proba_of_success'].extend(log_proba_of_success.tolist())
        # metrics['entropy'].extend(entropy.tolist())

    df['proba_of_success'] = metrics['proba_of_success']
    # df['entropy'] = metrics['entropy']

    return df

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries', type=Path, required=True)
    parser.add_argument('--model', type=str, default='paraphrase-MiniLM-L6-v2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    # load the model
    model = SentenceTransformer(args.model, device=args.device)

    # load the summaries
    df = parse_summaries(args.summaries)

    # embedd the text and the summary
    text_embeddings, summary_embeddings = embed_text_and_summaries(df, model)

    # compute the dot product between the text and the summary
    df = compute_dot_products(df, text_embeddings, summary_embeddings)

    # create the output directory
    args.output.mkdir(parents=True, exist_ok=True)

    path = args.output / f"{args.summaries.stem}.csv"

    # save the results
    df.to_csv(path, index=False)


if __name__ == '__main__':
    main()