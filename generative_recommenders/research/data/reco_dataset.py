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


from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import numpy as np

import torch

from generative_recommenders.research.data.dataset import DatasetV2
from generative_recommenders.research.data.preprocessor import get_common_preprocessors


@dataclass
class RecoDataset:
    max_sequence_length: int
    num_unique_items: int
    max_item_id: int
    all_item_ids: List[int]
    train_dataset: torch.utils.data.Dataset
    eval_dataset: torch.utils.data.Dataset
    text_embedding_dictmat: Optional[torch.Tensor] = None


def get_reco_dataset(
    dataset_name: str,
    text_embedding_model: str,
    max_sequence_length: int,
    chronological: bool,
) -> RecoDataset:
    if dataset_name in ["amzn23_office", "amzn23_game"]:

        dp = get_common_preprocessors(text_embedding_model)[dataset_name]
        train_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            shift_id_by=1,  # [0..n-1] -> [1..n]
            chronological=chronological,
        )
        eval_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            shift_id_by=1,  # [0..n-1] -> [1..n]
            chronological=chronological,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    # expected_max_item_id and item_features are not set for Amazon datasets.
    max_item_id = dp.expected_num_unique_items()
    all_item_ids = [x + 1 for x in range(max_item_id)]  # pyre-ignore [6]

    if text_embedding_model is None:
        text_embedding_dictmat = None
    else:
        text_embedding_path = f"tmp/{dataset_name}/item_text_embeddings_{text_embedding_model}.pt"
        text_embedding_dictmat = torch.load(text_embedding_path).to(torch.float32)


    return RecoDataset(
        max_sequence_length=max_sequence_length,
        num_unique_items=dp.expected_num_unique_items(),  # pyre-ignore [6]
        max_item_id=max_item_id,  # pyre-ignore [6]
        all_item_ids=all_item_ids,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        text_embedding_dictmat=text_embedding_dictmat,
    )
