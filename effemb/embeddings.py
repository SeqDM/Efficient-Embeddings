from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from transformers import AutoModel

import gin

SGPT_MODELS = {
    "small": "Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit",
    "medium": "Muennighoff/SGPT-1.3B-weightedmean-msmarco-specb-bitfit",
    "large": "Muennighoff/SGPT-2.7B-weightedmean-msmarco-specb-bitfit",
    "XL": "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit",

}

GPT2_MODELS = {
    "small": "gpt2",
    "medium": "gpt2-medium",
    "large": "gpt2-large",
    "XL": "gpt2-xl",
}

GPTNEO_MODELS = {
                "small": "EleutherAI/gpt-neo-125m",
                 "medium": "EleutherAI/gpt-neo-1.3B",
                 "large": "EleutherAI/gpt-neo-2.7B",
            }

class EmbeddingMethod:
    def __init__(self, method):
        self.method = method

    def get_embeddings(self, batch_tokens, last_hidden_state):
        # Get attn mask of shape [bs, seq_len, hid_dim]
        last_hidden_state = last_hidden_state.to("cpu")
        input_mask_expanded = (
            batch_tokens["attention_mask"]
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
            .to("cpu")
        )
        if self.method == "average":
            return self.get_embeddings_average(last_hidden_state,
                                                   input_mask_expanded)
        elif self.method == "sgpt":
            return self.get_embeddings_sgpt(last_hidden_state,
                                            input_mask_expanded)
        elif self.method == "last":
            return self.get_embeddings_last(last_hidden_state,
                                            batch_tokens)
        else:
            raise NotImplementedError

    def get_embeddings_average(self, last_hidden_state, input_mask_expanded):
        embeddings = (last_hidden_state * input_mask_expanded).mean(axis=1)
        return embeddings

    def get_embeddings_sgpt(self, last_hidden_state, input_mask_expanded):
        weights = (
            torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
            .to(last_hidden_state.device)
        )

        # Perform weighted mean pooling across seq_len:
        # bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(
            last_hidden_state * input_mask_expanded * weights, dim=1
        )
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)
        embeddings = sum_embeddings / sum_mask
        return embeddings

    def get_embeddings_last(self, last_hidden_state, batch_tokens):
                mask = batch_tokens["attention_mask"]
                # Create a new batch of masks with ones where the last one was
                # in the previous mask for each element in the batch

                # Find the indices of the last ones in each mask and set the
                # corresponding positions to 1
                mask = mask.flip(dims=(1,))
                first_occurences = mask.argmax(dim=1)
                last_occurences = mask.shape[1] - 1 - first_occurences
                first_ind = torch.arange(mask.shape[0])
                last_mask = torch.zeros_like(mask)
                last_mask[first_ind, last_occurences] = 1
                input_mask_expanded = (
                    last_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                ).to(last_hidden_state.device)
                embeddings = (
                    (last_hidden_state * input_mask_expanded).sum(axis=1)
                )
                return embeddings


class AbsEmbedding(ABC):
    @abstractmethod
    def encode(self, texts: List[str]):
        """
        encodes a batch of texts.
        :param texts: texts to be embedded
        """
        raise NotImplementedError


@gin.configurable
class TrainableEmbedding(AbsEmbedding):
    def __init__(self, model, tokenizer, device, method="average",
                 batch_scaling_factor=4):
        self.tokenizer = tokenizer
        self.model = model
        self.method = EmbeddingMethod(method)
        self.batch_scaling_factor = batch_scaling_factor
        self.device = device
        self.num_encoded_tokens = 0

    def get_num_encoded_tokens(self):
        return self.num_encoded_tokens

    def __call__(self, **kwargs):
        return self.forward(kwargs)

    def forward(self, batch):
        last_hidden_state = self.model(**batch,
           output_hidden_states=True, return_dict=True # TODO are these needed?
        ).last_hidden_state
        embeddings = self.method.get_embeddings(batch, last_hidden_state)

        self.num_encoded_tokens += batch["attention_mask"].sum()

        return embeddings

    def encode(self, texts: List[str], batch_size, **kwargs):
        result = None
        i = 0

        for k, v in kwargs.items():
            print(f"Unused argument to encode {k}={v}")

        # batch_size provided by the MTEB eval framework is typically under-optimal
        # by being too small; here we incease it for speeding up evaluation
        batch_size *= self.batch_scaling_factor

        while not i * batch_size >= len(texts):
            start_index = i * batch_size
            stop_index = min(len(texts), (i + 1) * batch_size)
            batch_tokens = self.tokenizer(texts[start_index:stop_index])
            batch_tokens.to(self.device)
            # Get the embeddings
            # Get hidden state of shape [bs, seq_len, hid_dim]
            last_hidden_state = self.model(
                **batch_tokens, output_hidden_states=True, return_dict=True
            ).last_hidden_state

            embeddings = self.method.get_embeddings(batch_tokens, last_hidden_state)
            if result is None:
                result = torch.zeros((len(texts), embeddings.shape[-1]))
            result[start_index:stop_index] = embeddings.to(self.device)
            i += 1

        return result.numpy()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save_pretrained(self, path):
        self.model.save_pretrained(path)

    def to(self, device):
        self.model.to(device)

@gin.configurable
class GPTEmbedding(TrainableEmbedding):
    def __init__(self, model_type="GPT2", model_version="small", method="sgpt"):
        if model_type == "GPT2":
            model_group = GPT2_MODELS
        elif model_type == "SGPT":
            model_group = SGPT_MODELS
        elif model_type == "GPTNEO":
            model_group = GPTNEO_MODELS
        else:
            raise NotImplementedError()

        model = AutoModel.from_pretrained(model_group[model_version]).eval()

        super(GPTEmbedding, self).__init__(model, tokenizer, method)
