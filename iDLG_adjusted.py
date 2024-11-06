import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from utils_model import compute_global_min_max, MinMaxNormalize, Dataset_from_Spectrogram, export_data, audio_mnist_dataset, LeNet_mnist, LeNet_urban

torch.manual_seed(0)

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

def main():
    dataset = 'audio_mnist'
    with_plots = False
    root_path = os.getcwd()
    data_path = os.path.join(root_path, 'data').replace('\\', '/')
    save_path = os.path.join(root_path, 'results/iDLG_%s'%dataset).replace('\\', '/')
    
    lr = 1
    num_dummy = 1
    Iteration = 300
    num_exp = 1000

    use_cuda = torch.cuda.is_available()
    # device = 'cuda' if use_cuda else 'cpu'
    device = 'cpu'

    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', save_path)

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)



    ''' load data '''
    if dataset == 'audio_mnist':
        # shape_img = (513, 92) # n_fft=1024
        shape_img = (257, 186)
        num_classes = 10
        channel = 1
        # hidden = 35604 # n_fft=1024
        hidden = 36660
        data_path = os.path.join(root_path, 'data/mnist/data_spec')
        images_all, labels_all, file_names = audio_mnist_dataset(data_path, shape_img)
        dst = Dataset_from_Spectrogram(images_all, np.asarray(labels_all, dtype=int)) # Load Dataset
        global_min, global_max = compute_global_min_max(dst) # Find min/max
        normalizer = MinMaxNormalize(global_min, global_max) # Normalize dataset
        transform = transforms.Compose([normalizer])
        dst = Dataset_from_Spectrogram(images_all, np.asarray(labels_all, dtype=int), transform=transform)

    elif dataset == 'urbansound':
        shape_img = (1025, 186)
        num_classes = 10
        channel = 1
        hidden = 144948
        data_path = os.path.join(root_path, 'data/audio/data_spec')
        images_all, labels_all, file_names = audio_mnist_dataset(data_path, shape_img)
        dst = Dataset_from_Spectrogram(images_all, np.asarray(labels_all, dtype=int))
        global_min, global_max = compute_global_min_max(dst)
        normalizer = MinMaxNormalize(global_min, global_max)
        transform = transforms.Compose([normalizer])
        dst = Dataset_from_Spectrogram(images_all, np.asarray(labels_all, dtype=int), transform=transform)

    else:
        exit('unknown dataset')




    ''' train DLG and iDLG '''
    for idx_net in range(num_exp):
        if dataset == 'audio_mnist':
            net = LeNet_mnist(channel=channel, hideen=hidden, num_classes=num_classes)
        elif dataset == 'urbansound':
            net = LeNet_urban(channel=channel, hideen=hidden, num_classes=num_classes)
        
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
            
            if with_plots:
                history = []
                history_iters = []

            losses = []
            mses = []
            train_iters = []
           
            print('lr =', lr)
            for iters in range(Iteration):
                
                if iters % int(Iteration / 20) == 0: #Track the progress of the training for plots
                    if with_plots:
                        dummy_data_log = dummy_data
                        history.append([dummy_data_log[imidx].clone().detach().cpu().numpy() for imidx in range(num_dummy)])
                        history_iters.append(iters)
                        
                def closure():
                    optimizer.zero_grad()
                    pred = net(dummy_data)
                    
                    if method == 'DLG':
                        dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
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


                if iters % int(Iteration / 20) == 0:
                    current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    print(current_time, iters, 'loss = %.8f, mse = %.8f' %(current_loss, mses[-1]))


                if current_loss < 0.00001: # converge
                    if with_plots:
                        dummy_data_log = dummy_data
                        history.append([dummy_data_log[imidx].clone().detach().cpu().numpy() for imidx in range(num_dummy)])
                        history_iters.append(iters)
                    break
            
                if with_plots:
                    for imidx in range(num_dummy):
                            plt.figure(figsize=(12, 8))
                            plt.subplot(3, 10, 1)
                            plt.imshow(gt_data[imidx][0].cpu(), cmap='gray')
                            for i in range(len(history)):
                                plt.subplot(3, 10, i + 2)
                                # Retrieve the image tensor for the current iteration and image index
                                img = history[i][imidx][0]
                                plt.imshow(img, cmap='gray')
                                plt.title('iter=%d' % (history_iters[i]))
                                plt.axis('off')
                            if method == 'DLG':
                                plt.savefig('%s/DLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                                plt.close()
                            elif method == 'iDLG':
                                plt.savefig('%s/iDLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                                plt.close()

        
            
            loss_iDLG = losses
            label_iDLG = label_pred.item()
            mse_iDLG = mses
            original_file_name = file_names[imidx_list[imidx]] 
            #Extract spectrogram from dummy data and save it
            if dataset == 'audio_mnist':
                save_path = "results_mat/iDLG_audio_mnist"
                export_data(normalizer.reverse(dummy_data.detach().cpu().numpy()),f'{save_path}/{original_file_name}')
            if dataset == 'urbansound':
                save_path = "results_mat/iDLG_urbansound"
                export_data(normalizer.reverse(dummy_data.detach().cpu().numpy()),f'{save_path}/{original_file_name}')



        print('imidx_list:', imidx_list)
        print('file name', file_names[imidx_list[0]])
        print('loss_iDLG:', loss_iDLG[-1])
        print('mse_iDLG:', mse_iDLG[-1])
        print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_iDLG:', label_iDLG)

        print('----------------------\n\n')

if __name__ == '__main__':
    main()

