import sys
import torchvision
import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np
from PIL import Image

class NonEpisodicCars(Dataset):
    name="Cars"
    task="cls"
    split_paths = {"train":"train", "test":"test", "valid": "val"}
    c = 3
    h = 84
    w = 84

    def __init__(self, data_root, split, transforms, rotation_labels=[0, 1, 2, 3], **kwargs):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        self.data_root = data_root
        self.split = {"train":"base", "val":"val", "valid":"val", "test":"novel"}[split]
        with open(os.path.join(self.data_root, "few_shot_lists", "%s.json" %self.split), 'r') as infile:
            self.metadata = json.load(infile)
        self.transforms = transforms
        self.rotation_labels = rotation_labels
        self.labels = np.array(self.metadata['image_labels'])
        label_map = {l: i for i, l in enumerate(sorted(np.unique(self.labels)))}
        self.labels = np.array([label_map[l] for l in self.labels])
        self.size = len(self.metadata["image_labels"])

    def next_run(self):
        pass

    def __getitem__(self, item):
        image = np.array(Image.open(self.metadata["image_names"][item]).convert("RGB"))
        images = self.transforms(image) * 2 - 1
        return images, int(self.labels[item])

    def __len__(self):
        return len(self.labels)