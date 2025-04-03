# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import abc

import torch

from generative_recommenders.research.modeling.initialization import truncated_normal


class EmbeddingModule(torch.nn.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class LocalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.reset_params()

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class CategoricalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_id_to_category_id: torch.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.register_buffer("_item_id_to_category_id", item_id_to_category_id)
        self.reset_params()

    def debug_str(self) -> str:
        return f"cat_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class ItemEmbeddingWithText(EmbeddingModule):

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        text_embedding_dim: int,
        text_embeddings: torch.Tensor,
    ) -> None:
        """
        Embedding module that combines learned item embeddings with precomputed textual embeddings.

        Args:
            num_items (int): Number of unique items.
            item_embedding_dim (int): Dimension of item ID embeddings.
            text_embedding_dim (int): Dimension of textual embeddings.
            text_embeddings (torch.Tensor): Precomputed item textual embeddings.
        """
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )

        self.register_buffer("_text_embeddings", text_embeddings)

        self._text_projection: torch.nn.Linear = torch.nn.Linear(
            text_embedding_dim, item_embedding_dim
        )

        self.reset_params()

    def debug_str(self) -> str:
        return f"text_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name or "_text_projection" in name:
                print(f"Initialize {name} as truncated normal: {params.data.size()} params")
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Retrieve combined embeddings for item IDs.

        Args:
            item_ids (torch.Tensor): Tensor of item IDs (batch_size, sequence_length).

        Returns:
            torch.Tensor: Combined item embeddings.
        """
        device = item_ids.device  # Ensure all tensors are on the same device

        # Move embeddings to the correct device
        item_embeds = self._item_emb(item_ids.to(device))  # Learned embeddings
        text_embeds = self._text_embeddings[item_ids.to(device)]  # Precomputed textual embeddings
        projected_text_embeds = self._text_projection(text_embeds.to(device))  # Projected textual embeddings

        return item_embeds + projected_text_embeds  # Summation for fusion

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim
