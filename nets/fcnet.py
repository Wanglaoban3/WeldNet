# 这个网络结构是论文《Automated defect classification of Aluminium 5083 TIG welding using HDR camera and neural networks》里
# 使用的，2022.8.4 Wangyi复现

import torch
import torch.nn as nn
from collections import OrderedDict

# config字典中包含了每种模型的配置，conv_config和maxpool_config子集中的两个元素分别代表kernel_size和stride
# linear_config代表卷积层后进入的第一个fc层输入通道数，planes代表第一个卷积层输出通道数
config = \
    {
        "model7":
            {"planes": 194800, "maxpool": None},
        "model8":
            {"planes": 48600, "maxpool": [2, 2]},
        "model9":
            {"planes": 21546, "maxpool": [3, 3]},
        "model10":
            {"planes": 7760, "maxpool": [5, 5]},
        "model11":
            {"planes": 1920, "maxpool": [10, 10]},
        "model12":
            {"planes": 480, "maxpool": [20, 20]},
    }


class Fcnet(nn.Module):
    def __init__(self, planes, maxpool, num_class):
        super(Fcnet, self).__init__()
        self.model = OrderedDict()
        if maxpool != None:
            self.model["maxpool"] = (nn.MaxPool2d(kernel_size=maxpool[0], stride=maxpool[1]))
        self.model["flatten"] = (nn.Flatten(start_dim=1))
        self.model["fc1"] = (nn.Linear(planes, 256))
        self.model["relu1"] = (nn.ReLU())
        self.model["fc2"] = (nn.Linear(256, 128))
        self.model["relu2"] = (nn.ReLU())
        self.model["fc3"] = (nn.Linear(128, num_class))
        self.model["softmax"] = (nn.Softmax(dim=1))
        self.model = nn.Sequential(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

# fcnet函数用来接收config字典并读取数据，生成模型。这样就可以直接导入config字典生成模型了
def fcnet(config, model_name, num_class):
    model_config = config[model_name]
    planes = model_config["planes"]
    maxpool_cofig = model_config["maxpool"]
    return Fcnet(planes, maxpool_cofig, num_class)


if __name__ == "__main__":
    model = fcnet(config, "model12", 6)
    x = torch.rand(1, 1, 400, 487)
    pred = model(x)
