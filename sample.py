import os
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from convert import Image2Tensor, CV2Tensor

class ExampleDataset(Dataset):
    # data loading
    def __init__(self, image_dir:str, trans):
        self.image_dir = image_dir
        self.image_list = os.listdir(image_dir)
        self.trans = trans
    # working for indexing
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_list[index])
        image = Image.open(image_path).convert('RGB')
        shape = image.size
        tensor = self.trans.toTensor(image)
        return tensor, np.array(shape)
    # return the length of our dataset
    def __len__(self):
        return len(self.image_list)

if __name__ == "__main__":
    img_dir = "sample/"
    trans_list = [transforms.Resize([640,640]), transforms.ToTensor()]
    trans = Image2Tensor(trans_list)

    dataloader = DataLoader(dataset=ExampleDataset(img_dir, trans), batch_size=2, shuffle=True)

    iter_data = iter(dataloader)
    tensor, shape = next(iter_data)
    print(tensor.size())
    for i, imt in enumerate(tensor):
        image = trans.toImage(imt)
        image = image.resize(shape[i])
        image.save(f"output/test_{str(i)}.jpg")
        print(image.size)
        