import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import os

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

def generate_tsne(array, root_path, file_name):
    tsne = TSNE(n_iter=2000).fit_transform(array)
    np.save(root_path+'/'+file_name+'.npy', tsne)
    return

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    root_path = 'visual_data_normal'
    # generate_preds(root_path, 'no_softmax')
    # datas = get_array(root_path+'/datas.npy')
    preds = get_array(root_path+'/no_softmax.npy')
    labels = get_array(root_path+'/labels.npy')
    # generate_tsne(preds, root_path, 'pred_tsne')
    tsne = get_array(root_path+'/pred_tsne.npy')

    fig = plt.figure(num=1, figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    tuli_label = ['Good weld', 'Burn through', 'Contamination', 'Lack of fusion', 'Misalignment', 'Lack of penetration']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for index in range(len(tuli_label)):
        plt.scatter(tsne[index==labels, 0], tsne[index==labels, 1], s=2, c=colors[index], label=tuli_label[index])

    # pca = PCA().fit_transform(preds)
    # plt.scatter(pca[:, 0], pca[:, 1], c=labels)
    # plt.colorbar()
    plt.legend()
    f = plt.gcf()
    f.savefig(root_path+'/test.png')
