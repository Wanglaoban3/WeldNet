import os
import argparse
import time
from openpyxl import Workbook
from copy import deepcopy
from utils.run import run



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default="C:/wrd/al5083/al5083")
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_path', type=str, default='logs_ss304/yuxunlian.pth')
    parser.add_argument('--epoch', type=int, default=25)
    parser.add_argument('--model_type', type=str, default='resnet18_attention')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--filename', type=str, default='')
    opt = parser.parse_args()
    fine_tune_layer = ['linear.weight', 'linear.bias']
    my_run = run(opt.root_path, opt.num_classes, opt.batch_size,
                 opt.model_type, opt.model_path, opt.pretrained, 'SGD')

    timeStamp = time.time()  # time.time()返回的是一个时间戳，以1970年1月1日到目前为止的秒数
    timeArray = time.localtime(timeStamp)  # localtime()是转化为年月日分秒的格式
    otherStyleTime = time.strftime("%Y_%m_%d %H_%M_%S", timeArray)  # strftime转化为字符串格式
    print('开始训练', opt.filename)

    max_test_acc = 0
    if opt.pretrained:
        my_run.fine_tune(fine_tune_layer)
    for epoch in range(0, opt.epoch):  #开始训练
        start_time = time.time()
        # my_run.train_one_epoch()
        my_run.eval_model()

        if my_run.epoch_results['test_acc'][epoch] > max_test_acc:
            max_test_acc = my_run.epoch_results['test_acc'][epoch]
            # best_weight = deepcopy(my_run.model)
            # best_weight = best_weight.cpu().state_dict()
        end_time = time.time()
        consume_time = round(end_time - start_time, 1)

        print(f'第{epoch}轮,花费{consume_time}秒,'
              f'训练集准确率、损失: {round(my_run.epoch_results["train_acc"][epoch], 3)}, {round(my_run.epoch_results["train_loss"][epoch], 5)},'
              f'测试集准确率、损失: {round(my_run.epoch_results["test_acc"][epoch], 3)}, {round(my_run.epoch_results["test_loss"][epoch], 5)},'
              f'最高, {round(max_test_acc, 3)}, 学习率: {round(my_run.schedulers[0].get_last_lr()[0], 5)}')
        my_run.epoch += 1

    file_name = [opt.filename, opt.model_type, str(max_test_acc), otherStyleTime]
    file_name = '_'.join(file_name)
    save_path = './logs_al/' + file_name
    os.makedirs(save_path, exist_ok=True)
    # torch.save(best_weight, save_path+'/best_weight.pth')
    wb = Workbook()
    ws = wb.active
    ws.append(['epoch', 'train_acc', 'train_loss', 'val_acc', 'val_loss'])
    for i in range(opt.epoch):
        ws.append([i+1, my_run.epoch_results["train_acc"][i], my_run.epoch_results["train_loss"][i],
                   my_run.epoch_results["test_acc"][i], my_run.epoch_results["test_loss"][i]])
    wb.save(save_path+'/datas.xlsx')