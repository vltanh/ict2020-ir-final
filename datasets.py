import csv

import torch
from torch.utils.data import Dataset
from torchvision import transforms as tvtf
from PIL import Image


class AIC2020Track2(Dataset):
    def __init__(self, root, path, train):
        self.train = train

        image_name, _, labels = zip(*list(csv.reader(open(path)))[1:])
        self.image_name = [root + '/' + image_path
                           for image_path in image_name]

        labels = list(map(int, labels))
        labels_set = set(labels)
        labels_mapping = {k: i for i, k in enumerate(labels_set)}
        
        labels = torch.tensor([x for x in labels])

        self.transform = tvtf.Compose([
            tvtf.Resize((224, 224)),
            tvtf.ToTensor(),
            tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        ])

        self.labels = labels
        self.data = self.image_name

    def __getitem__(self, index):
        image_path = self.image_name[index]
        im = Image.open(image_path)
        label = self.labels[index]
        return self.transform(im), label

    def __len__(self):
        return len(self.image_name)
