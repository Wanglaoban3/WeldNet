

import torch
import torch.nn as nn

# config字典中包含了每种模型的配置，conv_config和maxpool_config子集中的两个元素分别代表kernel_size和stride
# linear_config代表卷积层后进入的第一个fc层输入通道数，planes代表第一个卷积层输出通道数
model_config = \
    {
        "model1":
            {"planes": 16,
             "conv_config": [[5, 1], [5, 1], [5, 1], [5, 1]],
             "maxpool_config": [[5, 3], [5, 3], [5, 3], [5, 3]],
             "linear_config": 384
             },
        "model2":
            {"planes": 16,
             "conv_config": [[5, 1], [5, 1], [5, 1], [5, 1]],
             "maxpool_config": [[3, 2], [3, 2], [3, 2], [9, 9]],
             "linear_config": 2560
             },
        "model3":
            {"planes": 32,
             "conv_config": [[5, 2], [5, 2], [5, 2], [3, 2]],
             "maxpool_config": [[3, 2], [3, 2], [3, 2]],
             "linear_config": 512
             },
        "model4":
            {"planes": 16,
             "conv_config": [[3, 1], [3, 1], [3, 1], [3, 1]],
             "maxpool_config": [[5, 3], [5, 3], [5, 3], [5, 3]],
             "linear_config": 1024
             },
        "model5":
            {"planes": 16,
             "conv_config": [[3, 1], [3, 1], [3, 1], [3, 1]],
             "maxpool_config": [[3, 2], [3, 2], [3, 2], [9, 9]],
             "linear_config": 3840
             },
        "model6":
            {"planes": 16,
             "conv_config": [[3, 2], [3, 2], [3, 2], [3, 2]],
             "maxpool_config": [[3, 2], [3, 2], [3, 2]],
             "linear_config": 512
             }
    }


class dwConvnet(nn.Module):
    def __init__(self, planes, conv_config, maxpool_config, linear_config, num_class):
        super(dwConvnet, self).__init__()
        self.planes = planes
        self.block = []
        self.backbone = self.conv_block(conv_config, maxpool_config)
        self.fc1 = nn.Linear(linear_config, 256)
        self.fc_relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc_relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc_relu1(x)
        x = self.fc2(x)
        x = self.fc_relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def conv_block(self, conv_config, maxpool_config):
        for i in range(len(conv_config)):
            if i == 0:
                self.block.append(nn.Conv2d(1, self.planes, kernel_size=conv_config[i][0], stride=conv_config[i][1],
                                            groups=1))
                self.block.append(nn.ReLU())
                self.block.append(nn.MaxPool2d(maxpool_config[i][0], stride=maxpool_config[i][1]))
            else:
                self.block.append(nn.Conv2d(self.planes*(2**(i-1)), self.planes*(2**i), kernel_size=conv_config[i][0],
                                            stride=conv_config[i][1], groups=self.planes*(2**(i-1))))
                self.block.append(nn.ReLU())
                if i < len(maxpool_config):
                    self.block.append(nn.MaxPool2d(maxpool_config[i][0], stride=maxpool_config[i][1]))
        return nn.Sequential(*self.block)

# convnet函数用来接收config字典并读取数据，生成模型。这样就可以直接导入config字典生成模型了
def dwconvnet(model_name, num_class, config=model_config):
    config = config[model_name]
    planes = config["planes"]
    conv_config = config["conv_config"]
    maxpool_cofig = config["maxpool_config"]
    linear_config = config["linear_config"]
    return dwConvnet(planes, conv_config, maxpool_cofig, linear_config, num_class)

if __name__ == "__main__":
    model = dwconvnet("model6", 6)
    x = torch.rand(1, 1, 400, 487)
    pred = model(x)