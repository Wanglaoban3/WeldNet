import torch
import numpy as np
import os
import matplotlib.pyplot as plt


def generate_model():
    from torchvision.models import inception_v3
    model = inception_v3(aux_logits=False)
    model.fc = torch.nn.Linear(2048, 6)
    model.Conv2d_1a_3x3.conv = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
    stat_dict = torch.load("../logs/inception_v3_2022_12_23 22_02_55_ssl_0100迁移学习用/best_weight.pth")
    model.load_state_dict(stat_dict, strict=True)
    return model

def generate_dataset():
    from Datasets import my_dataset
    from torch.utils.data import DataLoader
    root_path = "C:/wrd/铝合金数据集/al5083"
    val_datapath = root_path + '/test'
    val_dataset = my_dataset(val_datapath, transform='val_trans')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    return val_dataloader

def generate_datas_labels(root_path):
    dataloader = generate_dataset()

    datas = np.zeros((len(dataloader), 487, 400))
    labels = np.zeros((len(dataloader), 1))
    for index, (x, label) in enumerate(dataloader):
        datas[index] = x.squeeze().numpy()
        labels[index] = label.numpy()
    datas = datas.reshape([len(dataloader), -1])
    labels = labels.squeeze()
    os.makedirs(root_path, exist_ok=True)
    np.save(root_path+'/datas.npy', datas)
    np.save(root_path+'/labels.npy', labels)
    print(datas.shape)  # (150,4)
    # 对应的标签有0,1,2三种
    print(labels.shape)  # (150,)
    return

def generate_preds(root_path, filename):
    model = generate_model()
    device = "cuda"
    if device == "cuda":
        model = model.cuda()
    model.eval()
    dataloader = generate_dataset()
    preds = np.zeros((len(dataloader), 6))
    with torch.no_grad():
        for index, (x, label) in enumerate(dataloader):
            if device == "cuda":
                x = x.cuda()
            pred = model(x)
            preds[index] = pred.detach().cpu().squeeze().numpy()
            # preds[index] = torch.softmax(pred, dim=1).detach().cpu().squeeze().numpy()

    preds = preds.reshape([len(dataloader), -1])
    os.makedirs(root_path, exist_ok=True)
    np.save(root_path+'/'+filename+'.npy', preds)
    # 对应的标签有0,1,2三种
    return

def get_array(file_path):
    array = np.load(file_path)
    return array



    # 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    import itertools
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.figure(figsize=(15, 12.5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(root_path+'/'+'normal_confuse_matrix.png')

if __name__ == "__main__":
    from sklearn.metrics import confusion_matrix as confusion_matrix_func

    root_path = 'confuse_matrix'
    # generate_preds(root_path, 'no_softmax')
    # datas = get_array(root_path+'/datas.npy')
    preds = get_array(root_path+'/normal_preds.npy')
    labels = get_array(root_path+'/labels.npy')
    y_pred = np.argmax(preds, axis=1)
    y_test = labels
    classes = ['Good weld', 'Burn through', 'Contamination', 'Lack of fusion', 'Misalignment', 'Lack of penetration']
    confusion_matrix_data = confusion_matrix_func(y_test, y_pred)
    plot_confusion_matrix(confusion_matrix_data, classes)


