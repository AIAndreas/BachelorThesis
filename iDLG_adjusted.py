import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pickle
import PIL.Image as Image
import tensorflow as tf
from torch.utils.data import DataLoader
import math


class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            nn.LayerNorm([12, 65, 25]),
            # nn.BatchNorm2d(12),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            nn.LayerNorm([12, 33, 13]),
            # nn.BatchNorm2d(12),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            nn.LayerNorm([12, 33, 13]),
            # nn.BatchNorm2d(12),
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


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())

def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    mean = 0.0
    std = 0.0
    nb_samples = 0
    sq_mean = 0.0
    total_pixels = 0
    n_channels = 1

    # for data, _ in loader:
        # batch_size, num_channels, height, width = data.shape
        # num_pixels = batch_size * height * width
        # mean += data.mean(axis=(2,3)).sum(0)
        # std += data.std(axis=(2,3)).sum(0)
        # nb_samples += 1

    mean = 0.0
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

class Dataset_from_Image(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs # img paths
        self.labs = labs # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, lab


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

def lfw_dataset(lfw_path, shape_img):
    images_all = []
    labels_all = []
    folders = os.listdir(lfw_path)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)
    

    transform = transforms.Compose([transforms.Resize(size=shape_img)])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst

def audio_mnist_dataset(data_path, shape_img):
    images_all = []
    labels_all = []
    files = os.listdir(data_path)
    for f in files:
        if f[-4:] == '.npy':
            images_all.append(os.path.join(data_path, f))
            labels_all.append(f[0])
    
    return images_all, labels_all


def adjust_learning_rate(optimizer, its, warmup_iterations=10):
    if its < warmup_iterations:
        lr = 0.5 # Increase LR over warmup epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = 0.05
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
def main():
    dataset = 'audio_mnist'
    root_path = os.getcwd()
    data_path = os.path.join(root_path, 'data').replace('\\', '/')
    save_path = os.path.join(root_path, 'results/iDLG_%s'%dataset).replace('\\', '/')
    
    lr = 1
    num_dummy = 1
    Iteration = 300
    num_exp = 1000

    use_cuda = torch.cuda.is_available()
    #device = 'cuda' if use_cuda else 'cpu'
    device = 'cpu'
    # tp = transforms.Compose([transforms.ToPILImage()])

    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', save_path)

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)



    ''' load data '''
    if dataset == 'MNIST':
        shape_img = (28, 28)
        num_classes = 10
        channel = 1
        hidden = 588
        dst = datasets.MNIST(data_path, download=True)

    elif dataset == 'cifar100':
        shape_img = (32, 32)
        num_classes = 100
        channel = 3
        hidden = 768
        dst = datasets.CIFAR100(data_path, download=False)


    elif dataset == 'lfw':
        shape_img = (32, 32)
        num_classes = 5749
        channel = 3
        hidden = 768
        lfw_path = os.path.join(root_path, 'data/lfw')
        dst = lfw_dataset(lfw_path, shape_img)

    elif dataset == 'audio_mnist':
        shape_img = (129, 51)
        num_classes = 10
        channel = 1
        hidden = 5148
        data_path = os.path.join(root_path, 'data/audioMNIST/data_spec')
        images_all, labels_all = audio_mnist_dataset(data_path, shape_img)
        dst = Dataset_from_Spectrogram(images_all, np.asarray(labels_all, dtype=int))
        # mean, std = compute_mean_std(dst)
        # transform = transforms.Compose([transforms.Normalize(mean=mean, std=std)])
        global_min, global_max = compute_global_min_max(dst)
        transform = transforms.Compose([MinMaxNormalize(global_min, global_max)])
        dst = Dataset_from_Spectrogram(images_all, np.asarray(labels_all, dtype=int), transform=transform)


    else:
        exit('unknown dataset')




    ''' train DLG and iDLG '''
    for idx_net in range(num_exp):
        net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes)
        net.apply(weights_init)

        print('running %d|%d experiment'%(idx_net, num_exp))
        net = net.to(device)
        idx_shuffle = np.random.permutation(len(dst))

        for method in ['iDLG']:
            print('%s, Try to generate %d images' % (method, num_dummy))

            criterion = nn.CrossEntropyLoss().to(device)
            imidx_list = []

            for imidx in range(num_dummy):
                idx = idx_shuffle[imidx]
                imidx_list.append(idx)
                tmp_datum = dst[idx][0].float().to(device)
                tmp_datum = tmp_datum.view(1, *tmp_datum.size())
                tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
                tmp_label = tmp_label.view(1, )
                print(tmp_datum.shape, tmp_label)
                if imidx == 0:
                    gt_data = tmp_datum
                    gt_label = tmp_label
                else:
                    gt_data = torch.cat((gt_data, tmp_datum), dim=0)
                    gt_label = torch.cat((gt_label, tmp_label), dim=0)

            print("gt_data_max:", torch.max(gt_data), "gt_data_min:", torch.min(gt_data))
            # compute original gradient
            out = net(gt_data)
            y = criterion(out, gt_label)
            dy_dx = torch.autograd.grad(y, net.parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))

            # generate dummy data and label
            dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
            dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)
            
            if method == 'DLG':
                optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
            elif method == 'iDLG':
                optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
                # predict the ground-truth label
                label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)
            
            history = []
            history_iters = []
            losses = []
            mses = []
            train_iters = []
            # print(torch.max(dummy_data), torch.min(dummy_data))
            print('lr =', lr)
            for iters in range(Iteration):
                
                # adjust_learning_rate(optimizer, iters)
                
                dummy_data_log = dummy_data
                if iters % int(Iteration / 20) == 0:
                    history.append([dummy_data_log[imidx].clone().detach().cpu().numpy() for imidx in range(num_dummy)])
                    # print(dummy_data[imidx].cpu())
                    history_iters.append(iters)
                    

                

                def closure():
                    optimizer.zero_grad()
                    pred = net(dummy_data)
                    # print(pred)
                    if method == 'DLG':
                        dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                        # print(dummy_loss)
                        # dummy_loss = criterion(pred, gt_label)
                    elif method == 'iDLG':
                        dummy_loss = criterion(pred, label_pred)
        
                    dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
    
                    grad_diff = 0

                    for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                        grad_diff += ((gx - gy) ** 2).sum()
                    grad_diff.backward()
                    
                    return grad_diff
                optimizer.step(closure)
                current_loss = closure().item()
                train_iters.append(iters)
                losses.append(current_loss)
                mses.append(torch.mean((dummy_data-gt_data)**2).item())

                # if math.isnan(current_loss): 
                #     for name, param in net.named_parameters():
                #         if param.grad is not None:
                #             print(f"Layer: {name}, Gradient: {param.grad.norm()}")
                #         else:
                #             print(f"Layer: {name} has no gradient")
                #     continue

                if iters % int(Iteration / 20) == 0:
                    current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    print(current_time, iters, 'loss = %.8f, mse = %.8f' %(current_loss, mses[-1]))


                if current_loss < 0.000001 and mses[-1] < 1: # converge
                    dummy_data_log = dummy_data
                    history.append([dummy_data_log[imidx].clone().detach().cpu().numpy() for imidx in range(num_dummy)])
                    history_iters.append(iters)
                    break

            # print("history_length:", len(history))
            # for imidx in range(num_dummy):
            #         plt.figure(figsize=(12, 8))
            #         plt.subplot(3, 10, 1)
            #         plt.imshow(gt_data[imidx][0].cpu(), cmap='gray')
            #         for i in range(len(history)):
            #             plt.subplot(3, 10, i + 2)
            #             # Retrieve the image tensor for the current iteration and image index
            #             # print(history[i][imidx])
            #             img = history[i][imidx][0]
            #             plt.imshow(img, cmap='gray')
            #             plt.title('iter=%d' % (history_iters[i]))
            #             plt.axis('off')
            #         if method == 'DLG':
            #             plt.savefig('%s/DLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
            #             plt.close()
            #         elif method == 'iDLG':
            #             plt.savefig('%s/iDLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
            #             plt.close()

            if method == 'DLG':
                loss_DLG = losses
                label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
                mse_DLG = mses
            elif method == 'iDLG':
                loss_iDLG = losses
                label_iDLG = label_pred.item()
                mse_iDLG = mses



        print('imidx_list:', imidx_list)
        print('loss_iDLG:', loss_iDLG[-1])
        print(, 'mse_iDLG:', mse_iDLG[-1])
        print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_DLG:', label_DLG, 'lab_iDLG:', label_iDLG)

        print('----------------------\n\n')

if __name__ == '__main__':
    main()

