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
import requests
from typing import Dict, Optional, Union

from urllib.request import urlretrieve
import gzip
import shutil

import pandas as pd
from datetime import datetime
from collections import defaultdict
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

class SteamDataProcessor(DataProcessor):
    def __init__(
        self,
        input_path: str,
        prefix: str,
        expected_num_unique_items: Optional[int] = None,
        text_embedding_model: Optional[str] = None,
    ) -> None:
        super().__init__(
            prefix,
            expected_num_unique_items=expected_num_unique_items,
            expected_max_item_id=None,
        )
        self._input_path = input_path
        self._prefix = prefix
        self._text_embedding_model = text_embedding_model
        self._include_text_embeddings = text_embedding_model is not None

    def download(self):
        # download and fix the misformatted json files
        out_dir = f"tmp/{self._prefix}"
        os.makedirs(out_dir, exist_ok=True)
        files = [
            ("https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz", "steam_reviews.json"),
            ("https://cseweb.ucsd.edu/~wckang/steam_games.json.gz", "steam_games.json"),
        ]
        for url, filename in files:
            raw_path = os.path.join(out_dir, filename)
            fixed_path = raw_path.replace(".json", "_fixed.json")
            gz_path = raw_path + ".gz"

            if not os.path.exists(raw_path):
                print(f"Downloading {url}")
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(gz_path, "wb") as f:
                        shutil.copyfileobj(r.raw, f)
                with gzip.open(gz_path, "rb") as f_in, open(raw_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(gz_path)

            if not os.path.exists(fixed_path):
                print(f"Fixing {raw_path}")
                fixed = []
                with open(raw_path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            line = line.replace("u'", "'").replace("'", '"')
                            line = line.replace("False", "false").replace("True", "true")
                            fixed.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                with open(fixed_path, "w", encoding="utf-8") as f:
                    json.dump(fixed, f, separators=(",", ":"))
                print(f"Saved fixed JSON to {fixed_path}")

    def parse_posted_date(self, posted_str):
        posted_str = posted_str.replace("Posted ", "").strip(". ")
        for fmt in ["%B %d, %Y", "%B %d", "%Y-%m-%d"]:
            try:
                return int(datetime.strptime(posted_str, fmt).timestamp())
            except Exception:
                continue
        return None

    def preprocess_rating(self) -> int:
        # 1. Ensure steam_reviews_fixed.json exists, download/fix if not
        reviews_fixed_path = f'tmp/{self._prefix}/steam_reviews_fixed.json'
        games_fixed_path = f'tmp/{self._prefix}/steam_games_fixed.json'
        if not os.path.exists(reviews_fixed_path) or not os.path.exists(games_fixed_path):
            print("Downloading and fixing Steam data files...")
            self.download()
        assert os.path.exists(reviews_fixed_path), f"{reviews_fixed_path} not found."
        assert os.path.exists(games_fixed_path), f"{games_fixed_path} not found."

        # 2. Create ratings DataFrame from reviews_fixed_path
        with open(reviews_fixed_path, 'r') as f:
            data = json.load(f)
        user_reviews = defaultdict(list)
        for review in data:
            user = review.get('username')
            item_id = review.get('product_id')
            posted = review.get('date', '')
            timestamp = self.parse_posted_date(posted)
            if user and item_id and timestamp:
                user_reviews[user].append((timestamp, item_id))
        all_usernames = sorted(user_reviews.keys())
        user2id = {uname: idx for idx, uname in enumerate(all_usernames)}
        rows = []
        for user, reviews in user_reviews.items():
            reviews.sort()
            uid = user2id[user]
            for ts, item_id in reviews:
                rows.append([uid, item_id, 1, ts])
        ratings_df = pd.DataFrame(rows, columns=['user_id', 'item_id', 'rating', 'timestamp'])

        # 3. Load item metadata from games_fixed_path
        with open(games_fixed_path, 'r') as f:
            metadata = json.load(f)
        item2meta = {}
        for entry in metadata:
            item_id = str(entry.get('id') or entry.get('item_id'))
            if not item_id:
                continue
            fields = []
            for key in ['app_name', 'publisher', 'developer', 'genres', 'tags', 'specs']:
                value = entry.get(key)
                if isinstance(value, list):
                    value = ', '.join(str(v) for v in value)
                if value:
                    fields.append(str(value))
            metadata_str = ' | '.join(fields)
            item2meta[item_id] = metadata_str

        # 4. Filter ratings so all item_ids exist in item2meta
        ratings_df = ratings_df[ratings_df['item_id'].astype(str).isin(item2meta.keys())]

        # 5. Filter item2meta so all item_ids are in filtered ratings
        filtered_item_ids = set(ratings_df['item_id'].astype(str).unique())
        item2meta = {item_id: meta for item_id, meta in item2meta.items() if item_id in filtered_item_ids}

        # 6. Filter ratings so only users with >=5 interactions remain
        user_counts = ratings_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= 5].index
        ratings_df = ratings_df[ratings_df['user_id'].isin(valid_users)]

        # Save filtered item2meta
        item2meta_path = f"tmp/{self._prefix}/item2meta.json"
        with open(item2meta_path, 'w') as f:
            json.dump(item2meta, f, indent=2, ensure_ascii=False)
        print(f"item2meta saved to {item2meta_path}")

        # Save mappings and id2item/id2meta based strictly on unique items in ratings_df
        unique_item_ids = sorted(ratings_df['item_id'].astype(str).unique())
        item2id = {item_id: idx for idx, item_id in enumerate(unique_item_ids)}
        id2item = unique_item_ids  # list: id -> item_id
        id2meta = [item2meta[item_id] for item_id in unique_item_ids]  # list: id -> meta
        mappings = {
            "user2id": user2id,
            "item2id": item2id,
            "id2item": id2item
        }
        mappings_path = f"tmp/{self._prefix}/mappings.json"
        with open(mappings_path, 'w') as f:
            json.dump(mappings, f, indent=2, ensure_ascii=False)
        print(f"{self._prefix} mappings (user2id, item2id, id2item) saved to {mappings_path}")

        # Remap item_id for sequence output
        ratings_df['remapped_item_id'] = ratings_df['item_id'].map(item2id)

        # Save id2meta to file
        id2meta_path = f"tmp/{self._prefix}/id2meta.json"
        with open(id2meta_path, 'w') as f:
            json.dump(id2meta, f, indent=2, ensure_ascii=False)
        print(f"{self._prefix} id2meta saved to {id2meta_path}")

        # Create text embeddings based on id2meta (must match unique items in ratings_df)
        embed_and_save_item_texts(
            id2meta,
            f"tmp/{self._prefix}/item_text_embeddings_{self._text_embedding_model}.pt",
            model_name=self._text_embedding_model,
        )
        print(f"{self._prefix} embeddings saved to tmp/{self._prefix}/item_text_embeddings_{self._text_embedding_model}.pt")
        # Save sequence format for SASRec/HSTU using remapped item_ids
        seq_df = (
            ratings_df.sort_values("timestamp")
            .groupby("user_id")
            .agg({
                "remapped_item_id": lambda x: ",".join(map(str, x)),
                "rating": lambda x: ",".join(map(str, x)),
                "timestamp": lambda x: ",".join(map(str, x)),
            })
            .reset_index()
        )
        seq_df.rename(columns={
            "remapped_item_id": "sequence_item_ids",
            "rating": "sequence_ratings",
            "timestamp": "sequence_timestamps",
        }, inplace=True)
        seq_path = f"tmp/{self._prefix}/sasrec_format.csv"
        seq_df.to_csv(seq_path, index=False, sep=",")
        print(f"{self._prefix} sequence-format saved to {seq_path}")

        print(f"{self._prefix} #data points: {ratings_df.shape[0]}")
        print(f"{self._prefix} #user: {ratings_df['user_id'].nunique()}")
        print(f"{self._prefix} #item: {ratings_df['item_id'].nunique()}")

        return ratings_df['item_id'].nunique()

    def process_meta(self, metadata_json_path: str, rating_csv_path: Optional[str] = None) -> dict:
        # Load game metadata, concatenate fields, and only keep item_ids in user review history
        with open(metadata_json_path, 'r') as f:
            data = json.load(f)

        if rating_csv_path is not None:
            sasrec_df = pd.read_csv(rating_csv_path)
            allowed_ids = set(sasrec_df['item_id'].astype(str))
        else:
            allowed_ids = None  # No restriction, include all items

        item2meta = {}
        for entry in data:
            item_id = str(entry.get('id') or entry.get('item_id'))
            if not item_id:
                continue
            if allowed_ids is not None and item_id not in allowed_ids:
                continue
            fields = []
            for key in ['app_name', 'publisher', 'developer', 'genres', 'tags', 'specs']:
                value = entry.get(key)
                if isinstance(value, list):
                    value = ', '.join(str(v) for v in value)
                if value:
                    fields.append(str(value))
            metadata = ' | '.join(fields)
            item2meta[item_id] = metadata

        return item2meta


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
            "amzn23_music": "Musical_Instruments",
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
    Union[AmazonDataProcessor, SteamDataProcessor],
]:
    amzn23_office_dp = AmazonDataProcessor(
        'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Office_Products.csv.gz',
        'tmp/Office_Products.csv',
        prefix="amzn23_office",
        text_embedding_model=text_embedding_model,
        expected_num_unique_items=77551,
    )
    amzn23_game_dp = AmazonDataProcessor(
        'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz',
        'tmp/Video_Games.csv',
        prefix="amzn23_game",
        text_embedding_model=text_embedding_model,
        expected_num_unique_items=25612,
    )
    amzn23_music_dp = AmazonDataProcessor(
        'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Musical_Instruments.csv.gz',
        'tmp/Musical_Instruments.csv',
        prefix="amzn23_music",
        text_embedding_model=text_embedding_model,
        expected_num_unique_items=24587,
    ) 
    steam_dp = SteamDataProcessor(
        'tmp/steam/steam_reviews_fixed.json',
        prefix="steam",
        text_embedding_model=text_embedding_model,
        expected_num_unique_items=11808,
    )
    return {
        "amzn23_office": amzn23_office_dp,
        "amzn23_game": amzn23_game_dp,
        "amzn23_music": amzn23_music_dp,
        "steam": steam_dp,
    }