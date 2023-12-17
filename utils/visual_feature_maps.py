from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from copy import deepcopy
import torch
import torch.nn as nn
from torchvision.models import resnet50, mobilenet_v3_small


class Resnet50(nn.Module):
    def __init__(self, class_num):
        super(Resnet50, self).__init__()
        self.backbone = nn.Sequential(*list(resnet50().children())[:-1])
        self.backbone[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.linear = nn.Linear(2048, class_num)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

# 导入数据
def get_image_info(image_dir):
    # 以RGB格式打开图像
    # Pytorch DataLoader就是使用PIL所读取的图像格式
    # 建议就用这种方法读取图像，当读入灰度图像时convert('')
    image_info = Image.open(image_dir)
    # 数据预处理方法
    image_transform = transforms.Compose([
        transforms.Resize([487, 400]),
        transforms.ToTensor(),
    ])
    image_info = image_transform(image_info)
    image_info = image_info.unsqueeze(0)
    return image_info

def generate_model(weight_path=None):
    # model = Resnet50(6)
    model = mobilenet_v3_small(progress=False)
    model.features[0][0] = torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[3] = torch.nn.Linear(in_features=1024, out_features=6, bias=True)

    if weight_path != None:
        stat_dict = torch.load(weight_path)
        model_dict = model.state_dict()
        for key, value in stat_dict.items():
            if key in model_dict and model_dict[key].shape == value.shape:
                model_dict[key] = value
        model.load_state_dict(model_dict)

    return model

# 获取第k层的特征图
def get_k_layer_feature_map(feature_extractor, k, x):
    with torch.no_grad():
        for index, layer in feature_extractor.named_children():
            x = layer(x)
            if k == index:
                return x


#  可视化特征图
def show_feature_map(feature_map, ori_img):
    size = ori_img.size()[2:]
    ori_img = ori_img.squeeze(0)
    feature_map = feature_map
    feature_map_sum = (feature_map.sum(dim=1) / feature_map.size(1)).squeeze().numpy()
    # 转PIL Image类型
    feature_map_sum = Image.fromarray(feature_map_sum)
    # 用torchvision.transform进行双线性插值，它只能对PIL Image类型操作
    T = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    feature_map_sum = T(feature_map_sum).squeeze().numpy()
    # 转PIL格式并保存
    T2 = transforms.ToPILImage()
    ori_img = T2(ori_img)

    plt.figure()

    save_path = "../logs_ss304/"

    plt.imsave(save_path+'fp.png', feature_map_sum)
    ori_img.save(save_path+'fp_ori.png')



if __name__ == '__main__':
    # 初始化图像的路径
    image_dir = "C:/wrd/铝合金数据集/al5083/train/170904-113012-Al 2mm-part1/frame_00228.png"
    model_path = '../logs_ss304/mobilv3_best_weight.pth'
    # 定义提取第几层的feature map
    k = 1
    # 导入Pytorch封装的AlexNet网络模型
    model = generate_model(model_path)

    # model = generate_model()
    # 是否使用gpu运算
    use_gpu = torch.cuda.is_available()
    use_gpu = False
    # 读取图像信息
    image_info = get_image_info(image_dir)
    # 判断是否使用gpu
    if use_gpu:
        model = model.cuda()
        image_info = image_info.cuda()
    # alexnet只有features部分有特征图
    # classifier部分的feature map是向量
    feature_extractor = model.features
    feature_map = get_k_layer_feature_map(feature_extractor, '10', image_info)
    show_feature_map(feature_map, image_info)