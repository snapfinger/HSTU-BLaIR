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
import logging
import os
import sys
import json
from typing import Dict, Optional, Union

from urllib.request import urlretrieve
import gzip
import shutil

import pandas as pd

from datasets import load_dataset
from .utils import *
from .item_text_embedding import embed_and_save_item_texts


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class DataProcessor:
    """
    This preprocessor does not remap item_ids. This is intended so that we can easily join other
    side-information based on item_ids later.
    """

    def __init__(
        self,
        prefix: str,
        expected_num_unique_items: Optional[int],
        expected_max_item_id: Optional[int],
    ) -> None:
        self._prefix: str = prefix
        self._expected_num_unique_items = expected_num_unique_items
        self._expected_max_item_id = expected_max_item_id

    @abc.abstractmethod
    def expected_num_unique_items(self) -> Optional[int]:
        return self._expected_num_unique_items

    @abc.abstractmethod
    def expected_max_item_id(self) -> Optional[int]:
        return self._expected_max_item_id

    @abc.abstractmethod
    def processed_item_csv(self) -> str:
        pass

    def output_format_csv(self) -> str:
        return f"tmp/{self._prefix}/sasrec_format.csv"

    def to_seq_data(
        self,
        ratings_data: pd.DataFrame,
        user_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if user_data is not None:
            ratings_data_transformed = ratings_data.join(
                user_data.set_index("user_id"), on="user_id"
            )
        else:
            ratings_data_transformed = ratings_data
        ratings_data_transformed.item_ids = ratings_data_transformed.item_ids.apply(
            lambda x: ",".join([str(v) for v in x])
        )
        ratings_data_transformed.ratings = ratings_data_transformed.ratings.apply(
            lambda x: ",".join([str(v) for v in x])
        )
        ratings_data_transformed.timestamps = ratings_data_transformed.timestamps.apply(
            lambda x: ",".join([str(v) for v in x])
        )
        ratings_data_transformed.rename(
            columns={
                "item_ids": "sequence_item_ids",
                "ratings": "sequence_ratings",
                "timestamps": "sequence_timestamps",
            },
            inplace=True,
        )
        return ratings_data_transformed

    def file_exists(self, name: str) -> bool:
        return os.path.isfile("%s/%s" % (os.getcwd(), name))


class AmazonDataProcessor(DataProcessor):
    def __init__(
        self,
        download_path: str,
        saved_name: str,
        prefix: str,
        expected_num_unique_items: Optional[int],
        text_embedding_model: Optional[str] = None,
    ) -> None:
        super().__init__(
            prefix,
            expected_num_unique_items=expected_num_unique_items,
            expected_max_item_id=None,
        )
        self._download_path = download_path
        self._saved_name = saved_name
        self._prefix = prefix
        self._text_embedding_model = text_embedding_model
        self._include_text_embeddings = text_embedding_model is not None

    def download(self) -> None:
        if not self.file_exists(self._saved_name):
            logging.info(f" Downloading dataset....")
            urlretrieve(self._download_path, self._saved_name)

        if self._saved_name.lower().endswith(".csv.gz"):
            logging.info(" Extracting .csv.gz file")
            output_path = self._saved_name[:-3]
            with gzip.open(self._saved_name, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            raise ValueError(f" Unknown archive type for file: {self._saved_name}")

    def preprocess_rating(self) -> int:
        self.download()

        ratings = pd.read_csv(
            self._saved_name.rstrip(".gz"),
            sep=",",
            names=["user_id", "item_id", "rating", "timestamp"],
            header=1 if "23" in self._prefix else None,
            low_memory=False,
        )

        if self._include_text_embeddings:
            item2meta = self.process_meta()
            ratings = ratings.apply(
                lambda row: filter_items_wo_metadata_df(row, item2meta), axis=1
            ).dropna()

        item_id_count = ratings["item_id"].value_counts()
        user_id_count = ratings["user_id"].value_counts()
        ratings = ratings[ratings["item_id"].isin(item_id_count[item_id_count >= 5].index)]
        ratings = ratings[ratings["user_id"].isin(user_id_count[user_id_count >= 5].index)]

        if self._include_text_embeddings:
            data_maps = remap_id(ratings)
            id2meta = {
                data_maps["item2id"][item]: text
                for item, text in item2meta.items()
                if item in data_maps["item2id"]
            }
            data_maps["id2meta"] = id2meta

            output_path = f'tmp/{self._prefix}/data_maps'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(data_maps, f)

            sorted_text = [id2meta[i] for i in range(len(id2meta))]
            embed_and_save_item_texts(
                sorted_text,
                f"tmp/{self._prefix}/item_text_embeddings_{self._text_embedding_model}.pt",
                model_name=self._text_embedding_model,
            )

            ratings["item_id"] = ratings["item_id"].map(lambda x: data_maps["item2id"][x])
            ratings["user_id"] = ratings["user_id"].map(lambda x: data_maps["user2id"][x])
        else:
            ratings["item_id"] = pd.Categorical(ratings["item_id"]).codes
            ratings["user_id"] = pd.Categorical(ratings["user_id"]).codes

        print(f"{self._prefix} #data points: {ratings.shape[0]}")
        print(
            f"{self._prefix} #user: {len(set(ratings['user_id'].values))}"
        )
        print(
            f"{self._prefix} #item: {len(set(ratings['item_id'].values))}"
        )

        num_unique_items = ratings["item_id"].nunique()
        ratings_group = ratings.sort_values(by="timestamp").groupby("user_id")

        seq_ratings_data = pd.DataFrame({
            "user_id": list(ratings_group.groups.keys()),
            "item_ids": ratings_group.item_id.apply(list).tolist(),
            "ratings": ratings_group.rating.apply(list).tolist(),
            "timestamps": ratings_group.timestamp.apply(list).tolist(),
        })
        seq_ratings_data = seq_ratings_data[seq_ratings_data["item_ids"].apply(len) >= 5]

        if not os.path.exists(f"tmp/{self._prefix}"):
            os.makedirs(f"tmp/{self._prefix}")

        seq_ratings_data = self.to_seq_data(seq_ratings_data)
        seq_ratings_data.sample(frac=1).reset_index(drop=True).to_csv(
            self.output_format_csv(), index=False, sep="," 
        )

        if self.expected_num_unique_items() is not None:
            assert self.expected_num_unique_items() == num_unique_items
            logging.info(f"{self.expected_num_unique_items()} unique items.")

        return num_unique_items

    def process_meta(self) -> Dict[str, str]:

        domain_map = {
            "amzn23_office": "Office_Products",
            "amzn23_game": "Video_Games",
        }

        domain = domain_map.get(self._prefix, "")
        if not domain:
            raise ValueError(f"Unknown domain prefix: {self._prefix}")

        meta_dataset = load_dataset(
            'McAuley-Lab/Amazon-Reviews-2023',
            f'raw_meta_{domain}',
            split='full',
            trust_remote_code=True
        ).map(clean_metadata, num_proc=16)

        return dict(zip(meta_dataset["parent_asin"], meta_dataset["cleaned_metadata"]))
    

def get_common_preprocessors(text_embedding_model: str) -> Dict[
    str,
    Union[AmazonDataProcessor],
]:
    amzn23_office_dp = AmazonDataProcessor(
        'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Office_Products.csv.gz',
        'tmp/Office_Products.csv.gz',
        prefix="amzn23_office",
        text_embedding_model=text_embedding_model,
        expected_num_unique_items=77551,
    )
    amzn23_game_dp = AmazonDataProcessor(
        'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz',
        'tmp/Video_Games.csv.gz',
        prefix="amzn23_game",
        text_embedding_model=text_embedding_model,
        expected_num_unique_items=25612,
    )
    return {
        "amzn23_office": amzn23_office_dp,
        "amzn23_game": amzn23_game_dp,
    }