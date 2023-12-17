import time
from torchvision.models.resnet import Bottleneck, ResNet, BasicBlock
from torchvision.models import MobileNetV2
from torchvision.models.mobilenetv3 import MobileNetV3, _mobilenet_v3_conf
import torch
import argparse
import numpy as np
from utils.Datasets import my_dataset
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from torchstat import stat
from openpyxl import Workbook

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_type', type=str, default='mobilenet_v3_large')
    parser.add_argument('--device', type=str, default='cpu')
    opt = parser.parse_args()

    FPS_datalist = []
    wb = Workbook()
    ws = wb.active
    ws.append(['model', 'cpu_fps', 'gpu_fps', 'params', 'Flops'])
    models = ['mobilenetv2', 'mobilenet_v3_small', 'mobilenet_v3_large', 'resnet18', 'resnet34', 'resnet50']
    devices = ['cpu', 'cuda']
    cpu_fps_list = []
    gpu_fps_list = []
    for opt.model_type in models:
        for opt.device in devices:

            if opt.model_type == 'resnet50':
                model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=opt.num_classes)   #实例化模型，加载权重并微调最后一层
                # opt.model_path = '../pretrained_weights/Resnet/resnet50-0676ba61.pth'
            elif opt.model_type == 'resnet18':
                model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=opt.num_classes)
                # opt.model_path = '../pretrained_weights/Resnet/resnet18-f37072fd.pth'
            elif opt.model_type == 'resnet34':
                model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=opt.num_classes)
                # opt.model_path = '../pretrained_weights/Resnet/resnet34-b627a593.pth'
            elif opt.model_type == 'mobilenetv2':
                model = MobileNetV2(num_classes=opt.num_classes)
                # opt.model_path = 'logs/mobilenetv2__2022_06_27 22_40_35 重划分数据集/best_weight.pth'
            elif opt.model_type == 'mobilenet_v3_small' or opt.model_type == 'mobilenet_v3_large':
                inverted_residual_setting, last_channel = _mobilenet_v3_conf(opt.model_type)
                model = MobileNetV3(inverted_residual_setting, last_channel, num_classes=opt.num_classes)
                # if opt.model_type == 'mobilenet_v3_large':
                #     opt.model_path = 'logs/mobilenet_v3_large__2022_06_28 13_52_07 重划分数据集/best_weight.pth'
                # else:
                #     opt.model_path = 'logs/mobilenet_v3_small__2022_06_28 10_12_37 重划分数据集/best_weight.pth'

            # model.load_state_dict(opt.model_path)


            model.eval()
            val_datapath = '铝合金数据集/al5083/test'
            val_dataset = my_dataset(val_datapath, transform='val_trans')
            val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)

            if opt.device == 'cpu':
                with torch.no_grad():
                    t_all = []
                    for i in range(3):
                        #由于每次测FPS都有小幅度波动，所以选择算3次取平均值
                        for x, _ in tqdm(val_dataloader):
                            t1 = time.time()
                            y = model(x)
                            t2 = time.time()
                            t_all.append(t2 - t1)
                print('model:', opt.model_type)
                print('average time:', np.mean(t_all) / 1)
                print('average fps:', 1 / np.mean(t_all))
                # print('fastest time:', min(t_all) / 1)
                # print('fastest fps:', 1 / min((t_all)))
                # print('slowest time:', max(t_all) / 1)
                # print('slowest fps:', 1 / max((t_all)))
                cpu_fps_list.append(1 / np.mean(t_all))

            else:
                t_all = []
                with torch.no_grad():
                    model = model.cuda()
                    for i in range(3):
                        for x, _ in tqdm(val_dataloader):
                            x = x.cuda()
                            torch.cuda.synchronize()
                            t1 = time.time()
                            y = model(x)
                            torch.cuda.synchronize()
                            t2 = time.time()
                            t_all.append(t2 - t1)
                print('model:', opt.model_type)
                print('average time:', np.mean(t_all) / 1)
                print('average fps:', 1 / np.mean(t_all))
                # print('fastest time:', min(t_all) / 1)
                # print('fastest fps:', 1 / min((t_all)))
                # print('slowest time:', max(t_all) / 1)
                # print('slowest fps:', 1 / max((t_all)))
                gpu_fps_list.append(1 / np.mean(t_all))
    for i in range(len(models)):
        ws.append((models[i], cpu_fps_list[i], gpu_fps_list[i]))
    wb.save('./logs/data_fps.xlsx')
