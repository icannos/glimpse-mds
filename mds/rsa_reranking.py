from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
from tqdm import tqdm


from rsasumm.rsa_reranker import RSAReranking

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--summaries", type=Path, default="")
    parser.add_argument("--output_dir", type=str, default="output")

    parser.add_argument("--filter", type=str, default=None)

    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def parse_summaries(path : Path) -> pd.DataFrame:
    summaries = pd.read_csv(path)

    # check if the dataframe has the right columns
    if not all(col in summaries.columns for col in ["id", "id_text", "id_candidate", "summary"]):
        raise ValueError("The dataframe must have columns ['id', 'id_text', 'id_candidate', 'summary']")

    return summaries

def reranking_rsa(summaries : pd.DataFrame, model, tokenizer, device):

    best_summaries = []
    best_bases = []
    for name, group in tqdm(summaries.groupby(["id"])):
        rsa_reranker = RSAReranking(model, tokenizer, device, group.summary.unique().tolist(), group.text.unique().tolist())
        best_rsa, best_base, speaker_df, listener_df, initial_listener, language_model_proba_df = rsa_reranker.rerank(t=3)

        group = group.set_index("summary")
        group_lines = group.loc[best_rsa]
        group_lines['speaker_proba'] = 0
        group_lines['listener_proba'] = 0
        group_lines['language_model_proba'] = 0
        group_lines['initial_listener_proba'] = 0

        group_lines = group_lines.reset_index()

        for i, (idx, line) in enumerate(group_lines.iterrows()):
            summary = line['summary']
            text = line['text']

            group_lines['speaker_proba'].loc[i] = speaker_df.loc[text, summary]
            group_lines['listener_proba'].loc[i] = listener_df.loc[text, summary]
            group_lines['language_model_proba'].loc[i] = language_model_proba_df.loc[text, summary]
            group_lines['initial_listener_proba'].loc[i] = initial_listener.loc[text, summary]


        group_lines["id"] = name
        best_summaries.append(group_lines)

        best_base_lines = group.loc[best_base]
        best_base_lines = best_base_lines.reset_index()

        best_base_lines['speaker_proba'] = 0
        best_base_lines['listener_proba'] = 0
        best_base_lines['language_model_proba'] = 0
        best_base_lines['initial_listener_proba'] = 0

        for i, (idx, line) in enumerate(best_base_lines.iterrows()):
            summary = line['summary']
            text = line['text']

            best_base_lines['speaker_proba'].loc[i] = speaker_df.loc[text, summary]
            best_base_lines['listener_proba'].loc[i] = listener_df.loc[text, summary]
            best_base_lines['language_model_proba'].loc[i] = language_model_proba_df.loc[text, summary]
            best_base_lines['initial_listener_proba'].loc[i] = initial_listener.loc[text, summary]


        best_base_lines["id"] = name
        best_bases.append(best_base_lines)

    best_summaries = pd.concat(best_summaries)
    best_bases = pd.concat(best_bases)

    return best_summaries, best_bases


def main():
    args = parse_args()

    if args.filter is not None:
        if args.filter not in args.summaries.stem:
            return

    # load the model and the tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = model.to(args.device)

    # load the summaries
    summaries = parse_summaries(args.summaries)

    # rerank the summaries
    best_summaries, bast_base = reranking_rsa(summaries, model, tokenizer, device=args.device)

    best_summaries['metadata/reranking_model'] = args.model_name
    best_summaries['metadata/rsa_iterations'] = 3

    bast_base['metadata/reranking_model'] = args.model_name
    bast_base['metadata/rsa_iterations'] = 3


    # save the summaries
    # make the output directory if it does not exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_dir) / f"{args.summaries.stem}-_-rsa_reranked.csv"
    output_path_base = Path(args.output_dir) / f"{args.summaries.stem}-_-base_reranked.csv"

    best_summaries.to_csv(output_path)
    bast_base.to_csv(output_path_base)


if __name__ == "__main__":
    main()
