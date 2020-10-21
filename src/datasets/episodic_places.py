import sys
import torchvision
import torch
from torch.utils.data import Dataset
from src.datasets.episodic_dataset import EpisodicDataset, FewShotSampler
import json
import os
import numpy as np
from PIL import Image
import numpy
import pickle,cv2

class EpisodicPlaces(EpisodicDataset):
    h = 84
    w = 84
    c = 3
    name="Places"
    task="cls"
    split_paths = {"train":"base", "val":"val", "valid":"val", "test":"novel"}
    def __init__(self, data_root, split, sampler, size, transforms):
        self.data_root = data_root
        self.split = "novel"#always read novel, not flexible here
        with open(os.path.join(data_root, "{}.pkl".format(self.split)), 'rb') as f:
            self.metadata = pickle.load(f)
        labels = np.array(self.metadata['image_labels'])
        label_map = {l: i for i, l in enumerate(sorted(np.unique(labels)))}
        labels = np.array([label_map[l] for l in labels])
        super().__init__(labels, sampler, size, transforms)
    
    def sample_images(self, indices):
        #return [np.array(Image.open(self.metadata['image_names'][i]).convert("RGB")) for i in indices]
        return [cv2.imdecode(self.metadata['image_names'][i], cv2.IMREAD_COLOR)[..., ::-1]for i in indices]

    def __iter__(self):
        return super().__iter__()

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    sampler = FewShotSampler(5, 5, 15, 0)
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                 torchvision.transforms.Resize((84,84)),
                                                 torchvision.transforms.ToTensor(),
                                                ])
    dataset = EpisodicPlaces("places","train", sampler, 10, transforms)
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x)
    if False:
        for batch in loader:
            from tools.plot_episode import plot_episode
            plot_episode(batch[0], classes_first=False)

