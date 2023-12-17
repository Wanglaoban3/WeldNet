from torch import nn as nn
from torch.utils.data import DataLoader
import torch
import sys
import os
sys.path.append(os.path.abspath('nets/'))
import utils.Datasets as Datasets


class run(object):
    def __init__(self, root_path, num_classes, batch_size, model_type, model_path, pretrained, optim):
        self.epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 训练前所准备的数据集，自己写好的dataset，用dataloader加载
        self.num_model = 3
        self.models = self._generate_model(num_classes, model_type, model_path, pretrained)
        self.optimizers, self.schedulers = self._generate_optim(optim)
        self.dataloader = self._generate_dataloader(root_path, batch_size)
        self.criteon = self._generate_criteon()
        self.bacth_results = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
        self.epoch_results = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    def _generate_criteon(self):
        criteon = nn.CrossEntropyLoss()
        if self.device.type == 'cuda':
            criteon = criteon.cuda()
        return criteon

    def _generate_optim(self, optim):
        optimizers, schedulers = [], []
        if optim == 'SGD':
            for i in range(self.num_model):
                optimizers.append(torch.optim.SGD(filter(lambda p: p.requires_grad, self.models[i].parameters()), lr=1e-1, momentum=0.9))
                schedulers.append(torch.optim.lr_scheduler.MultiStepLR(optimizers[i], milestones=[15], gamma=0.1))
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-3,
                                         betas=(0.9, 0.999), eps=1e-08)
        return optimizers, schedulers

    def _generate_model(self, num_classes, model_type, model_path, pretrained):
        from nets.Resnet import ResNet, BasicBlock, Bottleneck, Resnet_jianzhi
        from torchvision.models import MobileNetV2
        from torchvision.models.mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large
        from torchvision.models.inception import inception_v3
        from nets.convnet import convnet
        from nets.dwconvnet import dwconvnet
        models = []
        if model_type == 'resnet50':
            model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)   #实例化模型，加载权重并微调最后一层
        elif model_type == 'resnet10':
            model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)
        elif model_type == 'resnet18':
            for i in range(self.num_model):
                models.append(ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes))
        elif model_type == 'resnet18_jianzhi':
            for i in range(self.num_model):
                models.append(Resnet_jianzhi(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channel=1, distillation_ratio=2))
        elif model_type == 'resnet18_attention':
            for i in range(self.num_model):
                models.append(Resnet_jianzhi(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channel=1, attention=True))
        elif model_type == 'resnet34':
            model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

        elif model_type == 'mobilenet_v2':
            model = MobileNetV2(num_classes=num_classes)
            model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            nn.init.kaiming_normal_(model.features[0][0].weight, mode='fan_out')
        elif model_type == 'mobilenet_v3_small':
            model = mobilenet_v3_small(progress=False)
            model.features[0][0] = torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
            model.classifier[3] = torch.nn.Linear(in_features=1024, out_features=num_classes, bias=True)
            nn.init.normal_(model.classifier[3].weight, 0, 0.01)
            nn.init.zeros_(model.classifier[3].bias)
            nn.init.kaiming_normal_(model.features[0][0].weight, mode='fan_out')
        elif model_type == 'mobilenet_v3_large':
            model = mobilenet_v3_large(progress=False)
            model.features[0][0] = torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
            model.classifier[3] = torch.nn.Linear(in_features=1280, out_features=num_classes, bias=True)
            if pretrained is True and model_path == '':
                model_path = 'logs/mobilenet_v3_small__2022_11_25 10_08_15/best_weight.pth'
        elif model_type == 'model1':
            model = convnet(model_type, num_class=num_classes)
        elif model_type == 'dwmodel1':
            model = dwconvnet(model_type.split('dw')[1], num_class=num_classes)
        elif model_type == 'inception_v3':
            model = inception_v3(aux_logits=False)
            model.fc = nn.Linear(2048, num_classes)
            model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
            nn.init.kaiming_normal_(model.Conv2d_1a_3x3.conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.normal_(model.fc.weight, 0, 0.01)
            nn.init.constant_(model.fc.bias, 0)

        # if pretrained == True:
        #     model_dict = model.state_dict()
        #     pretrained_dict = torch.load(model_path)

            # import collections
            # dic = collections.OrderedDict()
            # for key, value in pretrained_dict.items():
            #     dic['backbone.'+key] = value
            # torch.save(dic, 'log_ss304/yuxunlian.pth')


            # for key, value in pretrained_dict.items():
            #     if key in model_dict and model_dict[key].shape == value.shape:
            #         model_dict[key] = value
            # model.load_state_dict(model_dict)
        if self.device.type == 'cuda':
            models = [model.cuda() for model in models]
        return models

    def _generate_dataloader(self, root_path, batch_size):
        import albumentations as A
        import albumentations.pytorch.transforms as T
        train_transform = A.Compose([
            A.RandomCrop(width=650, height=750),  # 随机裁剪
            A.HorizontalFlip(p=0.5),  # 水平翻转
            A.Rotate(limit=30),  # 随机旋转（正负45度）
            A.RandomBrightnessContrast(),  # 随机亮度和对比度调整
            A.GaussianBlur(),  # 高斯模糊
            A.GaussNoise(var_limit=(10.0, 50.0)),  # 随机高斯噪声
            A.Resize(width=400, height=487),  # 放缩
            A.ToFloat(max_value=255),
            T.ToTensorV2(),
        ])

        train_ann_path = root_path + '/train/train.json'
        train_dataset = Datasets.my_dataset(train_ann_path, transform=train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

        test_transform = A.Compose([A.Resize(width=400, height=487),
                                    A.ToFloat(max_value=255),
                                    T.ToTensorV2()])
        test_ann_path = root_path + '/test/test.json'
        test_dataset = Datasets.my_dataset(test_ann_path, transform=test_transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
        return {'train_loader': train_dataloader, 'test_loader': test_dataloader}

    def eval_model(self):
        [model.eval() for model in self.models]
        with torch.no_grad():
            it = 0
            for x, label in self.dataloader['test_loader']:
                it += 1
                if self.device.type == 'cuda':
                    x = x.cuda()
                    label = label.cuda()
                preds, losses = 0, 0
                for i in range(self.num_model):
                    pred = self.models[i](x)
                    loss = self.criteon(pred, label)
                    preds += pred
                    losses += loss
                logits_pred = torch.argmax(preds, dim=1)
                acc = (logits_pred == label).float().sum() / label.size(0)
                self.bacth_results['test_loss'].append((losses / self.num_model).item())
                self.bacth_results['test_acc'].append(acc.item())
        self.epoch_results['test_loss'].append(
            sum(self.bacth_results['test_loss'][self.epoch * it:(self.epoch+1) * it]) / it)
        self.epoch_results['test_acc'].append(
            sum(self.bacth_results['test_acc'][self.epoch * it:(self.epoch+1) * it]) / it)
        return

    def train_one_epoch(self):
        it = 0
        [model.train() for model in self.models]
        for i in range(self.num_model):
            for x, label in self.dataloader['train_loader']:
                it += 1
                if self.device.type == 'cuda':
                    x = x.cuda()
                    label = label.cuda()  # 两种写法，一种是用.cuda()，另一种是to(device)，将数据转移到指定设备上计算
                pred = self.models[i](x)
                loss = self.criteon(pred, label)
                logits_pred = torch.argmax(pred, dim=1)
                acc = (logits_pred==label).float().sum() / label.size(0)
                self.bacth_results['train_loss'].append(loss.item())
                self.bacth_results['train_acc'].append(acc.item())
                loss.backward()
                self.optimizers[i].step()
                self.optimizers[i].zero_grad()
            self.schedulers[i].step()
        self.epoch_results['train_loss'].append(
            sum(self.bacth_results['train_loss'][self.epoch*it: (self.epoch+1)*it]) / it)
        self.epoch_results['train_acc'].append(
            sum(self.bacth_results['train_acc'][self.epoch*it: (self.epoch+1)*it]) / it)
        return

    def fine_tune(self, names):
        for key, value in self.models.named_parameters():
            if key not in names:
                value.requires_grad = False

    def model_init(self):
        from torchvision.ops.misc import ConvNormActivation
        from torchvision.models.mobilenetv3 import InvertedResidual
        def init(weight):
            for m in weight.children():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0.001)
                    nn.init.constant_(m.bias, 0)
                # 也可以判断是否为conv2d，使用相应的初始化方式
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Sequential) or isinstance(m, ConvNormActivation) or isinstance(m, InvertedResidual):
                    init(m)
        init(self.model)