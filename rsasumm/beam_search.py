from typing import Tuple, Optional

import torch
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper


def compute_rsa_probas(
    logits: torch.Tensor, prior: torch.Tensor, rationality: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param logits: (world_size, num_beam, vocab_size)
    :param prior: (world_size, num_beam) for each beam the prior over the objects
    :param rationality: rationality parameter, the higher the more rational ie the more the speaker will try to adapt
    to the listener
    :return: S1, L1: (world_size, num_beam, vocab_size).
    S1[o, b, w] is the (log)probability of the word w given the object o and the current partial summary for the beam b
    L1[o, b, w] is the (log)probability of the object o given the word w and the current partial summary for the beam b
    """

    prod = logits + prior[..., None]

    L0 = torch.nan_to_num(torch.log_softmax(prod, dim=0), nan=-float("inf"))

    prod_s = logits + L0 * rationality

    S1 = torch.log_softmax(prod_s, dim=-1)
    S1 = torch.nan_to_num(S1, nan=-float("inf"))

    prod_l = logits + L0
    L1 = torch.log_softmax(prod_l, dim=0)
    L1 = torch.nan_to_num(L1, nan=-float("inf"))

    return S1, L1


def sample_from_probs(
    logits: torch.Tensor, num_beams: torch.Tensor, do_sample: bool, K: int = 10
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """

    :param logits: (num_beams, vocab_size) log proba for the next token only for the wanted object
    :param num_beams: number of beam to sample. (Can be different from the shape of logits since some beams might have
    finished earlier)
    :param do_sample: sample or use argmax
    :param K: number of samples to draw per beam to create the new population
    :return: idx_beam, idx_token, tokens_scores, the indices of the sampled tokens and their scores
    """

    vocab_size = logits.shape[-1]
    if do_sample:
        # sample from the probability distribution
        logits = logits.view(num_beams * logits.shape[-1])
        probs = torch.softmax(logits, dim=-1)
        samples = torch.multinomial(probs, num_samples=K * num_beams)

        # get the indices of the sampled tokens
        idx_beam, idx_token = samples // vocab_size, samples % vocab_size

        logits = logits.view(num_beams * vocab_size)

        tokens_scores = logits.gather(dim=-1, index=samples).squeeze(-1)

        return idx_beam, idx_token, tokens_scores

    else:
        # get the indices of the most probable tokens
        num_beams = logits.shape[0]
        vocab_size = logits.shape[-1]

        logits = logits.view(num_beams * vocab_size)
        scores, samples = logits.topk(2 * num_beams, dim=-1)

        idx_beam, idx_token = samples // vocab_size, samples % vocab_size

        tokens_scores = scores.squeeze(-1)

        return idx_beam, idx_token, tokens_scores


# Beam search RSA decoding
class RSAContextualDecoding:
    def __init__(self, model, tokenizer, device):
        """

        :param model:
        :param tokenizer:
        :param device:
        """

        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def fwd_pass(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Make a forward pass through the model to get the logits for the next tokens
        :param input_ids: (world_size, num_beams, input_length)
        :param decoder_input_ids: (world_size, num_beams, partial_target_length)
        :param attention_mask: (world_size, num_beams, input_length)
        :param decoder_attention_mask: (world_size, num_beams, partial_target_length)
        :return: logits: (world_size, num_beams, vocab_size)
        """
        with torch.no_grad():
            world_size, num_beams = input_ids.shape[0], decoder_input_ids.shape[1]

            input_ids = input_ids.view(world_size * num_beams, input_ids.shape[2]).to(self.device)
            attention_mask = attention_mask.view(
                world_size * num_beams, attention_mask.shape[2]
            ).to(self.device)

            decoder_input_ids = decoder_input_ids.view(
                world_size * num_beams, decoder_input_ids.shape[2]
            ).to(self.device)

            decoder_attention_mask = decoder_attention_mask.view(
                world_size * num_beams, decoder_attention_mask.shape[2]
            ).to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            logits = outputs.logits[..., -1, :]

            logits = logits.view(self.world_size, num_beams, logits.shape[-1])

            # return the probability of the next token when conditioned on the source text (world_size)
            # and the partial target text (num_beam)
            return logits

    def duplicate_and_align_input_ids(
        self,
        input_ids: torch.Tensor,
        input_ids_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_input_ids_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Duplicate the input_ids and decoder_input_ids to have all pairs of input_ids[i] and decoder_input_ids[j]
        It uses torch.repeat and torch.repeat_interleave to do get something like:
        a 1
        a 2
        a 3
        b 1
        b 2
        b 3
        ...
        :param input_ids: (world_size, input_length)
        :param decoder_input_ids: (num_beam, partial_target_length)
        :return: input_ids: (world_size, num_beam, input_length)
                 decoder_input_ids: (world_size, num_beam, partial_target_length)
                 aligned such that all pairs of input_ids[i] and decoder_input_ids[j] are present
        """

        num_beams = decoder_input_ids.shape[0]

        input_ids = input_ids.unsqueeze(1).repeat(1, num_beams, 1)
        input_ids_mask = input_ids_mask.unsqueeze(1).repeat(1, num_beams, 1)

        # repeat interleave
        decoder_input_ids = decoder_input_ids.repeat_interleave(self.world_size, dim=0)
        decoder_input_ids_mask = decoder_input_ids_mask.repeat_interleave(
            self.world_size, dim=0
        )

        decoder_input_ids = decoder_input_ids.view(self.world_size, num_beams, -1)
        decoder_input_ids_mask = decoder_input_ids_mask.view(
            self.world_size, num_beams, -1
        )

        # print(self.tokenizer.batch_decode(input_ids[0]))
        # print(self.tokenizer.batch_decode(decoder_input_ids[0]))

        return input_ids, input_ids_mask, decoder_input_ids, decoder_input_ids_mask

    def compute_rsa_probas(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        do_sample: bool = True,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        temperature: float = 1.0,
        rationality: float = 8.0,  # seems to be a good value
        process_logits_before_rsa: bool = True,
        beam_scores: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param input_ids: input_ids to the encoder/decoder model = source texts
        :param attention_mask: attention_mask to the encoder/decoder model
        :param decoder_input_ids: decoder ids / partial summaries
        :param decoder_attention_mask: attention mask for the decoder
        :param do_sample: are we planning on sampling the tokens or using argmax (to apply or not the logits processor)
        :param top_p: parameters for the logits processor top p
        :param top_k: parameters for the logits processor top k
        :param temperature: sampling temperature
        :param rationality: how rational is the speaker (higher means more rational)
        :param process_logits_before_rsa: should we apply the logits processor before or after the RSA computation
        :param beam_scores: (world_size, num_beams) the scores of the beams to be added to the logits
        :return: S1, L1: (world_size, num_beam, vocab_size).
        """

        # some sanity checks
        assert (top_p is None) or (
            top_k is None
        ), "top_p and top_k cannot be used together"
        assert ((top_p is not None) and (do_sample)) or (
            top_p is None
        ), "top_p can only be used with sampling"
        assert ((top_k is not None) and (do_sample)) or (
            top_k is None
        ), "top_k can only be used with sampling"

        # duplicate the input_ids and decoder_input_ids to have all pairs of input_ids[i] and decoder_input_ids[j]
        (
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
        ) = self.duplicate_and_align_input_ids(
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
        )

        logits = (
            self.fwd_pass(
                input_ids, decoder_input_ids, attention_mask, decoder_attention_mask
            )
            / temperature  # apply the temperature
        )

        logits = torch.nn.functional.log_softmax(logits, dim=-1)

        world_size = input_ids.shape[0]
        num_beams = decoder_input_ids.shape[1]

        logits = logits.view(world_size * num_beams, -1)

        if do_sample and process_logits_before_rsa:
            if top_p is not None:
                logits = TopPLogitsWarper(top_p=top_p)(input_ids=None, scores=logits)
            if top_k is not None:
                logits = TopKLogitsWarper(top_k=top_k)(input_ids=None, scores=logits)

        logits = logits.view(world_size, num_beams, -1)

        if beam_scores is not None:
            logits = logits + beam_scores[None, ..., None]

        # compute the RSA probabilities
        S1, L1 = compute_rsa_probas(logits, self.prior, rationality=rationality)
        logits = S1

        if do_sample and not process_logits_before_rsa:
            logits = logits.view(world_size * num_beams, -1)
            if top_p is not None:
                logits = TopPLogitsWarper(top_p=top_p)(input_ids=None, scores=logits)
            if top_k is not None:
                logits = TopKLogitsWarper(top_k=top_k)(input_ids=None, scores=logits)

            logits = logits.view(world_size, num_beams, -1)

        return logits, L1

    def generate(
        self,
        target_id: int,
        source_texts_ids: torch.Tensor,
        source_text_attention_mask: torch.Tensor,
        max_length: int = 100,
        num_beams: int = 8,
        do_sample=True,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        temperature: float = 1.0,
        rationality: float = 1.0,
        process_logits_before_rsa=True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param target_id: the id of the target object
        :param source_texts_ids: (world_size, input_length) the tokenized source texts
        :param source_text_attention_mask: (world_size, input_length) the attention mask for the source texts
        :param max_length: the maximum length to generate
        :param do_sample: are we sampling or using argmax
        :param top_p: parameters for the logits processor top p
        :param top_k: parameters for the logits processor top k
        :param temperature: sampling temperature
        :param rationality: how rational is the speaker (higher means more rational)
        :param process_logits_before_rsa: should we apply the logits processor before or after the RSA computation
        :return: decoder_input_ids : (num_beams, max_length) decoded sequences, beam_scores: (num_beams) the scores
        of the beams
        """

        self.num_beam = num_beams
        self.world_size = source_texts_ids.shape[0]

        self.prior = torch.ones((self.world_size, self.num_beam)).to(self.device) / self.world_size
        beam_scores = torch.zeros(self.num_beam).to(self.device)

        # initialize the decoder input ids
        decoder_input_ids = torch.full(
            (self.num_beam, 2),
            0,
            dtype=torch.long,
            device=self.device,
        )

        # initialize the decoder attention mask
        decoder_attention_mask = torch.ones_like(decoder_input_ids).to(self.device)

        new_beams = []
        finished_beams = []

        # run the beam search
        for t in range(max_length):
            # compute the RSA probabilities
            num_beams = decoder_input_ids.shape[0]

            S1, L1 = self.compute_rsa_probas(
                source_texts_ids,
                source_text_attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                rationality=rationality,
                beam_scores=beam_scores,
                process_logits_before_rsa=process_logits_before_rsa,
            )

            # sample from the probabilities
            idx_beam, idx_token, tokens_scores = sample_from_probs(
                S1[target_id].squeeze(), num_beams, do_sample
            )

            # create all the new beams

            new_beams = []

            for idx_t, idx_b, token_score in zip(idx_token, idx_beam, tokens_scores):
                new_beams.append(
                    (
                        decoder_input_ids[idx_b].tolist() + [idx_t.item()],
                        beam_scores[idx_b] + token_score.item(),
                        L1[:, idx_b, idx_t.item()],
                    )
                )

            # sort the beams
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)

            # keep only the best beams
            new_beams = new_beams[: self.num_beam]

            # check if the beams are finished
            _new_beams = []
            for beam in new_beams:
                if beam[0][-1] == self.tokenizer.eos_token_id:
                    finished_beams.append(beam)

                else:
                    _new_beams.append(beam)

            new_beams = _new_beams

            if len(new_beams) == 0:
                break

            # pad the beams
            max_beam_len = max(len(x[0]) for x in new_beams)
            new_beams = [
                (
                    x[0] + [self.tokenizer.pad_token_id] * (max_beam_len - len(x[0])),
                    x[1],
                    x[2],
                )
                for x in new_beams
            ]

            # update the beam scores
            beam_scores = torch.tensor([x[1] for x in new_beams]).to(self.device)

            # update the decoder input ids
            decoder_input_ids: torch.Tensor = torch.tensor(
                [x[0] for x in new_beams], device=self.device
            )

            # update the decoder attention mask based on pad tokens
            decoder_attention_mask = (
                decoder_input_ids != self.tokenizer.pad_token_id
            ).long()

            self.prior = torch.stack([x[2] for x in new_beams], dim=1).to(self.device)

            # self.prior = torch.ones((self.world_size, len(new_beams))) / self.world_size

        results = []

        # pad the beams
        max_beam_len = max(len(x[0]) for x in finished_beams + new_beams)
        for x in finished_beams + new_beams:
            results.append(
                (
                    x[0] + [self.tokenizer.pad_token_id] * (max_beam_len - len(x[0])),
                    x[1],
                    x[2],
                )
            )

        decoder_input_ids = torch.tensor([x[0] for x in results], device=self.device)

        beam_scores = torch.tensor([x[1] for x in results]).to(self.device)

        return decoder_input_ids, beam_scores
