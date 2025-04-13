# coding=utf-8
# Copyright 2024 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import random

import torch
# from parquet.parquet_dataset import CruiseParquetDataset
import os
import pyarrow.parquet as pq
from datasets import load_dataset

class CruiseParquetDataset:
    def __init__(self, 
                 data_path, 
                 rank=0, 
                 world_size=1, 
                 shuffle=True, 
                 repeat=True, 
                 verbose=False, 
                 buffer_size=1000, 
                 meta_data_path=None, 
                 state_path=None, 
                 num_workers=1):
        self.data_path = data_path
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.verbose = verbose
        self.buffer_size = buffer_size
        self.meta_data_path = meta_data_path
        self.state_path = state_path
        self.num_workers = num_workers
        self.dataset = self.load_data()

    def load_data(self):
        """Load data from the Parquet file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path {self.data_path} does not exist.")
        return pq.ParquetFile(self.data_path)

    def generate(self):
        """Generator function to yield data samples."""
        num_rows = self.dataset.metadata.num_rows
        indices = list(range(num_rows))
        
        if self.shuffle:
            random.shuffle(indices)

        for idx in indices:
            row = self.dataset.read_row_group(idx).to_pydict()
            yield row

        if self.repeat:
            self.generate()  # Repeat data if required

# class RefinedWebDataset(CruiseParquetDataset):
#     def __init__(self,
#                 data_path,
#                 rank: int = 0,
#                 world_size: int = 1,
#                 shuffle=True,
#                 repeat=True,
#                 buffer_size=1000,
#                 max_length=8000,
#                 num_workers=1,
#                 **kwargs
#                 ):
#         super().__init__(data_path, rank, world_size, shuffle, repeat, verbose=False, buffer_size=buffer_size, meta_data_path=None, state_path=None, num_workers=num_workers)
#         self.max_length = max_length

#     def __iter__(self):
#         for example in self.generate():
#             try:
#                 data, current_worker_hash, data_idx, seed = example
#                 text = data['content'].replace('\n', '')
#                 if len(text) > self.max_length:
#                     start_index = random.randint(0, len(text) - self.max_length - 1)
#                     selected_text = text[start_index:start_index + self.max_length]
#                 else:
#                     selected_text = text
#                 ret = {'input_ids': selected_text}
#                 yield ret

#             except Exception as e:
#                 # print('internal dataset iter error', e)
#                 continue

#     def collate_fn(self, batch):
#         batched = collections.defaultdict(list)
#         for data in batch:
#             for k, v in data.items():
#                 batched[k].append(v)
#         for k, v in batched.items():
#             if k not in ('key', 'input_ids', 'similarity'):
#                 batched[k] = torch.stack(v, dim=0)

#         return batched

class RefinedWebDataset:
    def __init__(self,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle=True,
                 repeat=True,
                 buffer_size=1000,
                 max_length=8000,
                 num_workers=1,
                 **kwargs):
        self.max_length = max_length
        self.shuffle = shuffle

        # Create a small set of sample data
        self.examples = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT")

        if self.shuffle:
            # random.shuffle(self.examples)
            self.examples = self.examples.shuffle(seed=42)

    def __len__(self):
        return len(self.examples["train"])


    def generate(self):
        for example in self.examples:
            # Simulate the additional fields
            yield example, "worker_hash", 0, random.randint(0, 1000)

    def __getitem__(self, index):
        return {'input_ids': self.examples["train"][index]["text"]}

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        return batched

if __name__ == '__main__':

    dataset = RefinedWebDataset('/mnt/bn/vgfm2/test_mlx/xavier/data/falcon-refinedweb/data/*.parquet', num_workers=10)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(dataset, batch_size=10,
                                  sampler=None, collate_fn=dataset.collate_fn,
                                  num_workers=10)
                                  # num_workers=0)
    for i, batch in enumerate(train_dataloader):
        print(len(batch['input_ids'][0]))
        import ipdb; ipdb.set_trace()
