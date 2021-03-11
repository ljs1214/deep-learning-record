from torchvision import transforms
import torch
import os
from torchvision.datasets import ImageFolder
import torchvision.datasets

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(48),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ]),
    'val': transforms.Compose([
        transforms.Scale(64),
        transforms.CenterCrop(48),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ]),
}

data_dir = './train_val_data/'
image_datasets = {x: ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x]) for x in ['train', 'val']}
dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                             batch_size=16,
                                             shuffle=True,
                                             num_workers=4) for x in ['train', 'val']}