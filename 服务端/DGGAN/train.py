# -*- coding: <encoding name> -*-
"""
训练生成模型(DGGAN)
"""

from __future__ import print_function, division
import torch.nn as nn
import torch
import os
import random
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
from DGGAN.dataset import GANDataset
from DGGAN.model import Generator, Discriminator
import matplotlib

################################################################################
# 相关配置
################################################################################
# Set random seed for reproducibility
manualSeed = 1
#manualSeed = random.randint(1, 10000) # use if you want new results
# print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# 选择在cpu或cuda运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 创建存储模型的目录
if not os.path.isdir('Checkpoint'):
    os.mkdir('Checkpoint')
    os.mkdir('Checkpoint/models')
    os.mkdir('Checkpoint/images')
# 设置演示动画限制大小
matplotlib.rcParams['animation.embed_limit'] = 2**128


################################################################################
# 训练模型
################################################################################
def train():
    """
    :return:
    """
    # 设置超参数
    PATH = 'Data/same_images/NV' # 训练图片存储目录
    LR = 1e-5  # 学习率
    EPOCH = 100  # 训练轮数
    BATCH_SIZE = 128  # Batch size
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    resume = False  # 是否断点训练
    workers = 2 # Number of workers for dataloader
    nz = 100  # Size of z latent vector (i.e. size of generator input)

    # 数据处理
    data_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=180),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # 加载数据
    dataset = GANDataset(PATH, transforms=data_transform, aug=False)
    # 定义trainloader
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=workers,
                              shuffle=True)

    # # 可视化部分训练图片
    # real_batch = next(iter(data_loader))
    # # real_batch = real_batch.mul(0.5).add(0.5)
    # plt.figure(figsize=(8, 8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(
    #     np.transpose(vutils.make_grid(real_batch.to(device)[:8], padding=2, normalize=True).cpu(), (1, 2, 0)))
    # plt.show()

    # 定义模型
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # 定义优化器
    # optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))
    # optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizerG = optim.SGD(netG.parameters(), lr=LR, weight_decay=1e-5)
    optimizerD = optim.SGD(netD.parameters(), lr=LR, weight_decay=1e-5)

    # 断点训练，加载模型权重
    if resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('Checkpoint'), 'Error: no Checkpoint directory found!'
        state = torch.load('Checkpoint/models/ckpt.pth')
        netG.load_state_dict(state['netG'])
        netD.load_state_dict(state['netD'])
        optimizerG.load_state_dict(state['optimG'])
        optimizerD.load_state_dict(state['optimD'])
        start_epoch = state['epoch']

    # 损失函数
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    # 训练模型
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(start_epoch, EPOCH):
        # For each batch in the dataloader
        for i, data in enumerate(data_loader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_batch = data.to(device)
            b_size = real_batch.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_batch).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 5 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, EPOCH, i, len(data_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 50 == 0) or ((epoch == EPOCH - 1) and (i == len(data_loader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                # 可视化生成图片
                fake = fake.mul(0.5).add(0.5)
                plt.figure(figsize=(8, 8))
                plt.axis("off")
                plt.title("Generating Images")
                plt.imshow(
                    np.transpose(vutils.make_grid(fake, padding=2, normalize=False, nrow=8), (1, 2, 0)))
                plt.show()
                # 保存生成图片
                img_list.append(vutils.make_grid(fake, padding=2, normalize=False, nrow=8))

            iters += 1
        # 保存模型
        state = {
            'netG': netG.state_dict(),
            'netD': netD.state_dict(),
            'optimG': optimizerG.state_dict(),
            'optimD': optimizerD.state_dict(),
            'epoch': epoch
        }
        torch.save(state, 'Checkpoint/models/ckpt.pth')

    # 绘制损失图
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    # 保存损失图
    plt.savefig('Checkpoint/images/loss_curve.png')
    # 显示损失图
    plt.show()

    # Visualization of G’s progression
    # %%capture
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())

    # Real Images vs. Fake Images
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(data_loader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[:64], padding=2, normalize=True, nrow=8), (1, 2, 0)))
    plt.show()

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()

################################################################################
# 函数入口
################################################################################
if __name__ == '__main__':
    # 训练生成模型
    train()