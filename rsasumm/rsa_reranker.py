from functools import cache
from typing import List

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm


def kl_divergence(p, q):
    """
    Compute the KL divergence between two distributions
    """
    return torch.nan_to_num(p * (p / q).log(), nan=0.0).sum(-1)


def jensen_shannon_divergence(p, q):
    """
    Compute the Jensen-Shannon divergence between two distributions
    """
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


class RSAReranking:
    """
    Rerank a list of candidates according to the RSA model.
    """

    def __init__(
            self,
            model,
            tokenizer,
            candidates: List[str],
            source_texts: List[str],
            batch_size: int = 32,
            rationality: int = 1,
            device="cuda",
    ):
        """
        :param model: hf model used to compute the likelihoods (supposed to be a seq2seq model), is S0 in the RSA model
        :param tokenizer:
        :param candidates: list of candidates summaries
        :param source_texts: list of source texts
        :param batch_size: batch size used to compute the likelihoods (can be high since we don't need gradients and
        it's a single forward pass)
        :param rationality: rationality parameter of the RSA model
        :param device: device used to compute the likelihoods
        """
        self.model = model
        self.device = device
        self.tokenizer = tokenizer

        self.candidates = candidates
        self.source_texts = source_texts

        self.batch_size = batch_size
        self.rationality = rationality

    def compute_conditionned_likelihood(
            self, x: List[str], y: List[str], mean: bool = True
    ) -> torch.Tensor:
        """
        Compute the likelihood of y given x

        :param x: list of source texts len(x) = batch_size
        :param y: list of candidates summaries len(y) = batch_size
        :param mean: average the likelihoods over the tokens of y or take the sum
        :return: tensor of shape (batch_size) containing the likelihoods of y given x
        """

        assert len(x) == len(y), "x and y must have the same length"

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        batch_size = len(x)

        x = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        y = self.tokenizer(y, return_tensors="pt", padding=True, truncation=True)
        # Concatenate the two inputs
        # Compute the likelihood of y given x

        x_ids = x.input_ids.to(self.device)
        y_ids = y.input_ids.to(self.device)

        logits = self.model(
            input_ids=x_ids,
            decoder_input_ids=y_ids,
            attention_mask=x.attention_mask.to(self.device),
            decoder_attention_mask=y.attention_mask.to(self.device),
        ).logits
        # Compute the likelihood of y given x

        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_ids = y_ids[..., 1:].contiguous()

        likelihood = -loss_fn(
            shifted_logits.view(-1, shifted_logits.size(-1)), shifted_ids.view(-1)
        )

        likelihood = likelihood.view(batch_size, -1).sum(-1)
        if mean:
            likelihood /= (y_ids != self.tokenizer.pad_token_id).float().sum(-1)

        return likelihood

    def score(self, x: List[str], y: List[str], **kwargs):
        return self.compute_conditionned_likelihood(x, y, **kwargs)

    def likelihood_matrix(self) -> torch.Tensor:
        """
        :return: likelihood matrix : (world_size, num_candidates), likelihood[i, j] is the likelihood of
        candidate j being a summary for source text i.
        """
        likelihood_matrix = torch.zeros(
            (len(self.source_texts), len(self.candidates))
        ).to(self.device)

        pairs = []
        for i, source_text in enumerate(self.source_texts):
            for j, candidate in enumerate(self.candidates):
                pairs.append((i, j, source_text, candidate))

        # split the pairs into batches
        batches = [
            pairs[i: i + self.batch_size]
            for i in range(0, len(pairs), self.batch_size)
        ]

        for batch in tqdm(batches):
            # get the source texts and candidates
            source_texts = [pair[2] for pair in batch]
            candidates = [pair[3] for pair in batch]

            # compute the likelihoods
            with torch.no_grad():
                likelihoods = self.score(
                    source_texts, candidates, mean=True
                )

            # fill the matrix
            for k, (i, j, _, _) in enumerate(batch):
                likelihood_matrix[i, j] = likelihoods[k].detach()

        return likelihood_matrix

    @cache
    def S(self, t):
        if t == 0:
            return self.initial_speaker_probas
        else:
            listener = self.L(t - 1)
            prod = listener * self.rationality # + self.initial_speaker_probas.sum(0, keepdim=True)
            return torch.log_softmax(prod, dim=-1)

    @cache
    def L(self, t):
        speaker = self.S(t)
        return torch.log_softmax(speaker, dim=-2)

    def mk_listener_dataframe(self, t):
        self.initial_speaker_probas = self.likelihood_matrix()

        initial_listener_probas = self.L(0)

        # compute consensus
        uniform_distribution_over_source_texts = torch.ones_like(
            initial_listener_probas
        ) / len(self.source_texts)

        initital_consensuality_score = (
                torch.exp(initial_listener_probas)
                * (
                        initial_listener_probas - torch.log(uniform_distribution_over_source_texts)
                )
        ).sum(0).cpu().numpy()

        initital_consensuality_score = pd.Series(initital_consensuality_score, index=self.candidates)

        initial_listener_probas = initial_listener_probas.cpu().numpy()

        initial_listener_probas = pd.DataFrame(initial_listener_probas)
        initial_listener_probas.index = self.source_texts
        initial_listener_probas.columns = self.candidates

        initial_speaker_probas = self.S(0).cpu().numpy()
        initial_speaker_probas = pd.DataFrame(initial_speaker_probas)
        initial_speaker_probas.index = self.source_texts
        initial_speaker_probas.columns = self.candidates

        listener_df = pd.DataFrame(self.L(t).cpu().numpy())

        consensuality_scores = (
                torch.exp(self.L(t))
                * (self.L(t) - torch.log(uniform_distribution_over_source_texts))
        ).sum(0).cpu().numpy()

        consensuality_scores = pd.Series(consensuality_scores, index=self.candidates)

        S = self.S(t).cpu().numpy()
        speaker_df = pd.DataFrame(S)

        # add the source texts and candidates as index

        listener_df.index = self.source_texts
        speaker_df.index = self.source_texts

        listener_df.columns = self.candidates
        speaker_df.columns = self.candidates

        return listener_df, speaker_df, initial_listener_probas, initial_speaker_probas, initital_consensuality_score, consensuality_scores

    def rerank(self, t=1):
        """
        return the best summary (according to rsa) for each text
        """
        (
            listener_df,
            speaker_df,
            initial_listener_proba,
            initial_speaker_proba,
            initital_consensuality_score,
            consensuality_scores,
        ) = self.mk_listener_dataframe(t=t)
        best_rsa = speaker_df.idxmax(axis=1).values
        best_base = initial_listener_proba.idxmax(axis=1).values

        return (
            best_rsa,
            best_base,
            speaker_df,
            listener_df,
            initial_listener_proba,
            initial_speaker_proba,
            initital_consensuality_score,
            consensuality_scores,
        )


class RSARerankingEmbedder(RSAReranking):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_embeddings(self, x: List[str], y: List[str], **kwargs):
        model_kwargs = kwargs.get("model_kwargs")

        # shape: (batch_size, embedding_dim)
        x_embeddings = self.model.encode(x, **model_kwargs)
        y_embeddings = self.model.encode(y, **model_kwargs)

        # dot product between the embeddings : shape (batch_size)
        dot_products = (x_embeddings * y_embeddings).sum(-1)

        return dot_products

    def score(self, x: List[str], y: List[str], **kwargs):
        return self.compute_embeddings(x, y, **kwargs)

