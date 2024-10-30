import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

class LeNet_mnist(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet_mnist, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            nn.LayerNorm([12, 65, 25]),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            nn.LayerNorm([12, 33, 13]),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            nn.LayerNorm([12, 33, 13]),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out  
    

class LeNet_urban(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet_urban, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            nn.LayerNorm([12,513,93]),            
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            nn.LayerNorm([12,257,47]),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            nn.LayerNorm([12,257,47]),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out  


def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    mean = 0.0
    std = 0.0
    sq_mean = 0.0
    total_pixels = 0

    for data, _ in loader:
        # data shape: [batch_size, num_channels, height, width]
        batch_size, num_channels, height, width = data.shape

        # Reshape to [batch_size, num_channels, height * width]
        data = data.view(batch_size, num_channels, -1)

        # Sum over batch and spatial dimensions (height * width)
        mean += data.sum(dim=[0, 2])
        sq_mean += (data ** 2).sum(dim=[0, 2])

        # Update total number of pixels per channel
        total_pixels += height * width * batch_size

    # Compute global mean and std
    mean /= total_pixels
    sq_mean /= total_pixels
    std = torch.sqrt(sq_mean - mean ** 2)

    # mean /= nb_samples
    # std /= nb_samples
    return mean, std

def compute_global_min_max(dataset, batch_size=64, num_workers=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    global_min = float('inf')
    global_max = float('-inf')

    for data, _ in loader:
        current_min = data.min().item()
        current_max = data.max().item()
        if current_min < global_min:
            global_min = current_min
        if current_max > global_max:
            global_max = current_max

    return global_min, global_max

class MinMaxNormalize:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor):
        return (tensor - self.min_val) / (self.max_val - self.min_val + 1e-8)
    
    def reverse(self, normalized_tensor):
        return normalized_tensor * (self.max_val - self.min_val + 1e-8) + self.min_val


class Dataset_from_Spectrogram(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs # img paths
        self.labs = labs # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        spec = np.load(self.imgs[idx])
        spec = np.expand_dims(spec, axis=0)
        spec = torch.from_numpy(spec).float()
        if self.transform:
            spec = self.transform(spec)
        return spec, lab

def audio_mnist_dataset(data_path, shape_img):
    images_all = []
    labels_all = []
    file_names = []
    files = os.listdir(data_path)
    for f in files:
        if f[-4:] == '.npy':
            images_all.append(os.path.join(data_path, f))
            labels_all.append(f[0])
            file_names.append(f)
    return images_all, labels_all, file_names

def export_data(data, dir):
    np.save(dir, data)
