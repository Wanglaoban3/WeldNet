import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR

# 定义模型
class SimCLR(nn.Module):
    def __init__(self):
        super(SimCLR, self).__init__()

        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # 投影头
        self.projection_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )

    # 对两个样本进行处理
    def forward(self, x1, x2):
        z1 = self.feature_extractor(x1)
        z2 = self.feature_extractor(x2)

        # 拼接在一起
        z = torch.cat([z1, z2], dim=0)

        # 投影头
        p = self.projection_head(z.squeeze().squeeze())

        # 计算相似度
        z1, z2 = torch.split(z, z.size(0) // 2, dim=0)
        p1, p2 = torch.split(p, p.size(0) // 2, dim=0)
        sim = (F.normalize(z1, p=2, dim=1) * F.normalize(z2, p=2, dim=1)).sum(dim=1)

        return p1, p2, sim

# 定义数据集类
class ImageDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        img1 = transforms.ToTensor()(img)
        img2 = transforms.RandomApply([transforms.RandomHorizontalFlip(), transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)], p=0.5)(img)
        img2 = transforms.ToTensor()(img2)
        return img1, img2

    def __len__(self):
        return len(self.dataset)


# 定义训练函数
def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data1, data2) in enumerate(train_loader):
        data1, data2 = data1.to(device), data2.to(device)
        optimizer.zero_grad()
        output1, output2, sim = model(data1, data2)
        loss = (-sim).exp().mean().log()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

if __name__ == "__main__":

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    train_set = ImageDataset(train_set)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

    # 定义设备和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimCLR().to(device)

    # 训练模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    for epoch in range(1, 51):
        train(model, device, train_loader, optimizer, epoch, log_interval=50)
        scheduler.step()

    #在这个代码示例中，我们定义了 `SimCLR` 模型，并在 `forward` 方法中对两个输入样本进行特征提取和投影头操作，并计算相似度。通过定义 `ImageDataset` 类来处理两个随机增强样本，使用 `CIFAR-10` 数据集进行训练。使用 Adam 优化器和学习率衰减调整训练过程。最后，我们训练了 50 个 epochs。

