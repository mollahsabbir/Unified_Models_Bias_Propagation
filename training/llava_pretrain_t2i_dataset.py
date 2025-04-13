import collections
from typing import Any, Callable, Optional

import torch
from torchvision.datasets.folder import DatasetFolder, default_loader
from training.utils import image_transform
from torch.utils.data import Dataset
import torch
import collections
from datasets import load_dataset
from PIL import Image
import os

class LLaVAPretrainT2I(Dataset):
    def __init__(self, image_root, split="train", image_size=256, max_number_of_items=364100):
        # if multi_resolutions is true, then image_size will be a list of resolutions
        
        dataset = load_dataset("liuhaotian/LLaVA-Pretrain", data_files="blip_laion_cc_sbu_558k_meta.json")

        if split not in dataset:
            raise ValueError(f"Invalid split '{split}'. Available splits: {list(dataset_dict.keys())}")
        self.dataset = dataset[split]  # Select the desired split
        self.dataset = self.dataset.select(range(min(max_number_of_items, len(self.dataset))))
        self.image_size = image_size
        self.transform = image_transform  # Define your image transform function
        self.image_root = image_root
        print(f"Llava pretrain {split} dataset loaded. Total samples: {len(self.dataset)}.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            # Retrieve image and label
            sample = self.dataset[idx]  # Index into the selected split
            image_path = sample["image"]
            image = Image.open(os.path.join(self.image_root, image_path))

            input_ids = sample["blip_caption"]

            
            # Apply transforms to the image
            image = self.transform(image, resolution=self.image_size)                
            return {'images': image, 'input_ids': input_ids}

        except Exception as e:
            print(f"Error at index {idx}: {e}")
            if idx + 1 < len(self.dataset):
                return self.__getitem__(idx + 1)  # Skip to the next sample on error
            else:
                raise IndexError("Index out of range while handling exceptions.")

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('input_ids'):
                batched[k] = torch.stack(v, dim=0)

        return batched