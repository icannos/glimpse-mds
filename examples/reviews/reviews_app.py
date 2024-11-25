import math
from typing import List, Tuple

import nltk
import numpy as np
import seaborn as sns

import sys, os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rsasumm.rsa_reranker import RSAReranking
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

MODEL = "facebook/bart-large-cnn"

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)


latex_template = r"""
    \begin{subfigure}[b]{0.48\textwidth}
        \resizebox{\textwidth}{!}{
            \begin{coloredbox}{darkgray}{Review 1}
            [REVIEW 1]
            
\end{coloredbox}}
    \end{subfigure}
        \begin{subfigure}[b]{0.48\textwidth}
        \resizebox{\textwidth}{!}{
        \begin{coloredbox}{darkgray}{Review 2}
        [REVIEW 2]
\end{coloredbox}}
    \end{subfigure}
        \begin{subfigure}[b]{0.48\textwidth}
        \resizebox{\textwidth}{!}{
        \begin{coloredbox}{darkgray}{Review 3}
        [REVIEW 3]
\end{coloredbox}}
    \end{subfigure}
    """

EXAMPLES = [
    "The paper gives really interesting insights on the topic of transfer learning. It is well presented and the experiment are extensive. I believe the authors missed Jane and al 2021. In addition, I think, there is a mistake in the math.",
    "The paper gives really interesting insights on the topic of transfer learning. It is well presented and the experiment are extensive. Some parts remain really unclear and I would like to see a more detailed explanation of the proposed method.",
    "The paper gives really interesting insights on the topic of transfer learning. It is not well presented and lack experiments. Some parts remain really unclear and I would like to see a more detailed explanation of the proposed method.",
]


def make_colored_text_to_latex(scored_texts : List[Tuple[str, float]]):
    """
    Make a latex string from a list of scored texts.
    """

    # cast scores between 0 and 1
    scores = np.array([score for _, score in scored_texts])
    scores = (scores - scores.min()) / (scores.max() - scores.min())

    # make color map in hex
    cmap = sns.diverging_palette(250, 30, l=50, center="dark", as_cmap=True)
    hex_colors = [cmap(score)[0:3] for score in scores]
    # make html color string
    hex_colors = [",".join([str(round(x, 2)) for x in color]) for color in hex_colors]
    # make latex string
    latex_string = ""
    for (text, score), hex_color in zip(scored_texts, hex_colors):
        latex_string += "\\textcolor[rgb]{" + str(hex_color) + "}{" + text + "} "

    return latex_string




def summarize(text1, text2, text3, iterations, rationality=1.0):
    # get sentences for each text

    text1_sentences = nltk.sent_tokenize(text1)
    text2_sentences = nltk.sent_tokenize(text2)
    text3_sentences = nltk.sent_tokenize(text3)


    # remove empty sentences
    text1_sentences = [sentence for sentence in text1_sentences if sentence != ""]
    text2_sentences = [sentence for sentence in text2_sentences if sentence != ""]
    text3_sentences = [sentence for sentence in text3_sentences if sentence != ""]

    sentences = list(set(text1_sentences + text2_sentences + text3_sentences))

    rsa_reranker = RSAReranking(
        model,
        tokenizer,
        candidates=sentences,
        source_texts=[text1, text2, text3],
        device="cpu",
        rationality=rationality,
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
    ) = rsa_reranker.rerank(t=iterations)

    # apply exp to the probabilities
    speaker_df = speaker_df.applymap(lambda x: math.exp(x))

    text_1_summaries = speaker_df.loc[text1][text1_sentences]
    text_1_summaries = text_1_summaries / text_1_summaries.sum()

    text_2_summaries = speaker_df.loc[text2][text2_sentences]
    text_2_summaries = text_2_summaries / text_2_summaries.sum()

    text_3_summaries = speaker_df.loc[text3][text3_sentences]
    text_3_summaries = text_3_summaries / text_3_summaries.sum()

    # make list of tuples
    text_1_summaries = [(sentence, text_1_summaries[sentence]) for sentence in text1_sentences]
    text_2_summaries = [(sentence, text_2_summaries[sentence]) for sentence in text2_sentences]
    text_3_summaries = [(sentence, text_3_summaries[sentence]) for sentence in text3_sentences]

    # normalize consensuality scores between -1 and 1

    consensuality_scores = (consensuality_scores - (consensuality_scores.max() - consensuality_scores.min()) / 2) / (consensuality_scores.max() - consensuality_scores.min()) / 2
    consensuality_scores_01 = (consensuality_scores - consensuality_scores.min()) / (consensuality_scores.max() - consensuality_scores.min())


    most_consensual = consensuality_scores.sort_values(ascending=True).head(3).index.tolist()
    least_consensual = consensuality_scores.sort_values(ascending=False).head(3).index.tolist()

    most_consensual = [(sentence, consensuality_scores[sentence]) for sentence in most_consensual]
    least_consensual = [(sentence, consensuality_scores[sentence]) for sentence in least_consensual]

    text_1_consensuality = consensuality_scores.loc[text1_sentences]
    text_2_consensuality = consensuality_scores.loc[text2_sentences]
    text_3_consensuality = consensuality_scores.loc[text3_sentences]

    # rescale between -1 and 1
    # text_1_consensuality = (text_1_consensuality - (text_1_consensuality.max() - text_1_consensuality.min()) / 2) / (text_1_consensuality.max() - text_1_consensuality.min()) / 2
    # text_2_consensuality = (text_2_consensuality - (text_2_consensuality.max() - text_2_consensuality.min()) / 2) / (text_2_consensuality.max() - text_2_consensuality.min()) / 2
    # text_3_consensuality = (text_3_consensuality - (text_3_consensuality.max() - text_3_consensuality.min()) / 2) / (text_3_consensuality.max() - text_3_consensuality.min()) / 2

    text_1_consensuality = [(sentence, text_1_consensuality[sentence]) for sentence in text1_sentences]
    text_2_consensuality = [(sentence, text_2_consensuality[sentence]) for sentence in text2_sentences]
    text_3_consensuality = [(sentence, text_3_consensuality[sentence]) for sentence in text3_sentences]

    fig1 = plt.figure(figsize=(20, 10))
    ax = fig1.add_subplot(111)
    sns.heatmap(
        listener_df,
        ax=ax,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        cbar=False,
        annot_kws={"size": 10},
    )
    ax.set_title("Listener probabilities")
    ax.set_xlabel("Candidate sentences")
    ax.set_ylabel("Source texts")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    fig1.tight_layout()

    fig2 = plt.figure(figsize=(20, 10))
    ax = fig2.add_subplot(111)
    sns.heatmap(
        speaker_df,
        ax=ax,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        cbar=False,
        annot_kws={"size": 10},
    )
    ax.set_title("Speaker probabilities")
    ax.set_xlabel("Candidate sentences")
    ax.set_ylabel("Source texts")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    fig2.tight_layout()

    latex_text_1 = make_colored_text_to_latex(text_1_summaries)
    latex_text_2 = make_colored_text_to_latex(text_2_summaries)
    latex_text_3 = make_colored_text_to_latex(text_3_summaries)

    text_1_consensuality_ = consensuality_scores_01.loc[text1_sentences]
    text_2_consensuality_ = consensuality_scores_01.loc[text2_sentences]
    text_3_consensuality_ = consensuality_scores_01.loc[text3_sentences]

    text_1_consensuality_ = [(sentence, text_1_consensuality_[sentence]) for sentence in text1_sentences]
    text_2_consensuality_ = [(sentence, text_2_consensuality_[sentence]) for sentence in text2_sentences]
    text_3_consensuality_ = [(sentence, text_3_consensuality_[sentence]) for sentence in text3_sentences]

    latex_text_1_consensuality = make_colored_text_to_latex(text_1_consensuality_)
    latex_text_2_consensuality = make_colored_text_to_latex(text_2_consensuality_)
    latex_text_3_consensuality = make_colored_text_to_latex(text_3_consensuality_)
    
    latex = latex_template.replace("[REVIEW 1]", latex_text_1)
    latex = latex.replace("[REVIEW 2]", latex_text_2)
    latex = latex.replace("[REVIEW 3]", latex_text_3)


    return text_1_summaries, text_2_summaries, text_3_summaries, text_1_consensuality, text_2_consensuality, text_3_consensuality, most_consensual, least_consensual, fig1, fig2, latex


# make gradiot highlightedText component


iface = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Textbox(lines=10, value=EXAMPLES[0]),
        gr.Textbox(lines=10, value=EXAMPLES[1]),
        gr.Textbox(lines=10, value=EXAMPLES[2]),
        gr.Number(value=1, label="Iterations"),
        gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=1.0, label="Rationality"),
    ],
    outputs=[
        gr.Highlightedtext(
            show_legend=True,
            label="Uniqueness score for each sentence in text 1",
        ),
        gr.Highlightedtext(
            show_legend=True,
            label="Uniqueness score for each sentence in text 2",
        ),
        gr.Highlightedtext(
            show_legend=True,
            label="Uniqueness score for each sentence in text 3",
        ),
        gr.Highlightedtext(
            show_legend=True,
            label="Consensuality score for each sentence in text 1",

        ),
        gr.Highlightedtext(
            show_legend=True,
            label="Consensuality score for each sentence in text 2",
        ),
        gr.Highlightedtext(
            show_legend=True,
            label="Consensuality score for each sentence in text 3",
        ),
        gr.Highlightedtext(
            show_legend=True,
            label="Most consensual sentences",

        ),
        gr.Highlightedtext(
            show_legend=True,
            label="Least consensual sentences",
        ),
        gr.Plot(
            label="Listener probabilities",
        ),
        gr.Plot(
            label="Speaker probabilities",
        ),

        gr.Textbox(lines=10, label="Latex Consensuality scores"),



    ],
    title="RSA Summarizer",
    description="Summarize 3 texts using RSA",
)

iface.launch(share=True)
