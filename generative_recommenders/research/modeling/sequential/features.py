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

from typing import Dict, NamedTuple, Optional, Tuple

import torch


class SequentialFeatures(NamedTuple):
    # (B,) x int64. Requires past_lengths[i] > 0 \forall i.
    past_lengths: torch.Tensor
    # (B, N,) x int64. 0 denotes valid ids.
    past_ids: torch.Tensor
    # (B, N, D) x float.
    past_embeddings: Optional[torch.Tensor]
    # Implementation-specific payloads.
    # e.g., past timestamps, past event_types (e.g., clicks, likes), etc.
    past_payloads: Dict[str, torch.Tensor]


def movielens_seq_features_from_row(
    row: Dict[str, torch.Tensor],
    device: int,
    max_output_length: int,
) -> Tuple[SequentialFeatures, torch.Tensor, torch.Tensor]:
    historical_lengths = row["history_lengths"].to(device)  # [B]
    historical_ids = row["historical_ids"].to(device)  # [B, N]
    historical_ratings = row["historical_ratings"].to(device)
    historical_timestamps = row["historical_timestamps"].to(device)
    target_ids = row["target_ids"].to(device).unsqueeze(1)  # [B, 1]
    target_ratings = row["target_ratings"].to(device).unsqueeze(1)
    target_timestamps = row["target_timestamps"].to(device).unsqueeze(1)

    # Retrieve historical text embeddings (if available)
    historical_text_embeddings = (
        row["historical_text_embeddings"].to(device) if "historical_text_embeddings" in row else None
    )  # [B, N, D_text]

    # Retrieve target text embeddings (if available)
    target_text_embeddings = (
        row["target_text_embeddings"].to(device) if "target_text_embeddings" in row else None
    )  # [B, 1, D_text]
    
    if max_output_length > 0:
        B = historical_lengths.size(0)
        historical_ids = torch.cat(
            [
                historical_ids,
                torch.zeros(
                    (B, max_output_length), dtype=historical_ids.dtype, device=device
                ),
            ],
            dim=1,
        )
        historical_ratings = torch.cat(
            [
                historical_ratings,
                torch.zeros(
                    (B, max_output_length),
                    dtype=historical_ratings.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        historical_timestamps = torch.cat(
            [
                historical_timestamps,
                torch.zeros(
                    (B, max_output_length),
                    dtype=historical_timestamps.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        historical_timestamps.scatter_(
            dim=1,
            index=historical_lengths.view(-1, 1),
            src=target_timestamps.view(-1, 1),
        )

        if historical_text_embeddings is not None:
            D_text = historical_text_embeddings.size(-1)
            padding_text_embeddings = torch.zeros((B, max_output_length, D_text), dtype=torch.float32, device=device)
            historical_text_embeddings = torch.cat([historical_text_embeddings, padding_text_embeddings], dim=1)

    features = SequentialFeatures(
        past_lengths=historical_lengths,
        past_ids=historical_ids,
        past_embeddings=None,
        past_payloads={
            "timestamps": historical_timestamps,
            "ratings": historical_ratings,
            "text_embeddings": historical_text_embeddings if historical_text_embeddings is not None else torch.zeros_like(historical_ids, dtype=torch.float32),
        },
    )
    return features, target_ids, target_ratings
