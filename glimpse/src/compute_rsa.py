from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
from tqdm import tqdm

from pickle import dump

import sys, os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rsasumm.rsa_reranker import RSAReranking


DESC = """
Compute the RSA matrices for all the set of multi-document samples and dump these along with additional information in a pickle file.
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--summaries", type=Path, default="summaries/Test.csv")
    parser.add_argument("--output_dir", type=str, default="output")

    parser.add_argument("--filter", type=str, default=None)

    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def parse_summaries(path: Path) -> pd.DataFrame:
    summaries = pd.read_csv(path)

    # check if the dataframe has the right columns
    if not all(
        col in summaries.columns for col in ["index", "id", "text", "gold", "summary", "id_candidate"]
    ):
        raise ValueError(
            "The dataframe must have columns ['index', 'id', 'text', 'gold', 'summary', 'id_candidate']"
        )

    return summaries


def compute_rsa(summaries: pd.DataFrame, model, tokenizer, device):
    results = []
    for name, group in tqdm(summaries.groupby(["id"])):
        rsa_reranker = RSAReranking(
            model,
            tokenizer,
            device=device,
            candidates=group.summary.unique().tolist(),
            source_texts=group.text.unique().tolist(),
            batch_size=32,
            rationality=3,
        )
        (
            best_rsa,
            best_base,
            speaker_df,
            listener_df,
            initial_listener,
            language_model_proba_df,
            initial_consensuality_scores,
            consensuality_scores,
        ) = rsa_reranker.rerank(t=2)

        gold = group['gold'].tolist()[0]

        results.append(
            {
                "id": name,
                "best_rsa": best_rsa,
                "best_base": best_base,
                "speaker_df": speaker_df,
                "listener_df": listener_df,
                "initial_listener": initial_listener,
                "language_model_proba_df": language_model_proba_df,
                "initial_consensuality_scores": initial_consensuality_scores,
                "consensuality_scores": consensuality_scores,
                "gold": gold,
                "rationality": 3,
                "text_candidates" : group
            }
        )

    return results


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
    results = compute_rsa(summaries, model, tokenizer, args.device)
    results = {"results": results}

    results["metadata/reranking_model"] = args.model_name
    results["metadata/rsa_iterations"] = 3

    results["metadata/reranking_model"] = args.model_name
    results["metadata/rsa_iterations"] = 3

    # save the summaries
    # make the output directory if it does not exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_dir) / f"{args.summaries.stem}-_-r3-_-rsa_reranked-{args.model_name.replace('/', '-')}.pk"
    output_path_base = (
        Path(args.output_dir) / f"{args.summaries.stem}-_-base_reranked.pk"
    )

    with open(output_path, "wb") as f:
        dump(results, f)


if __name__ == "__main__":
    main()
