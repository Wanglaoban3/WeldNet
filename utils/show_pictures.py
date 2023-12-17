import numpy as np
import matplotlib.pyplot as plt
import os

class data_show():
    def __init__(self, file_path, save_path=None, show=True):
        if type(file_path) == str:
            self.data = np.loadtxt(open(file_path, 'r'), delimiter=',')
        if type(file_path) == tuple:
            self.data = []
            self.data_names = []
            for i in file_path:
                self.data.append(np.loadtxt(open(i, 'r'), delimiter=','))
                self.data_names.append(os.path.split(i)[-1].split('_file.csv')[0])
        self.draw(self.data)
        if show == True:
            self.show()
        plt.imsave()

    def draw(self, data):
        plt.figure(dpi=130)  # 设置绘图窗口的大小
        plt.rcParams['font.sans-serif'] = 'SimHei'  # 中文显示
        plt.rcParams['axes.unicode_minus'] = False

        if type(data) != list:
            steps = int(len(self.data))
            x = range(0, steps)
            plt.plot(x, data)
        steps = int(len(self.data[0]))
        x = range(0, steps)
        title = str()
        for id, i in enumerate(data):
            plt.plot(x, i)
            if id + 1 == len(self.data_names):
                title += self.data_names[id]
                break
            title += self.data_names[id] + ', '
        plt.legend(self.data_names, loc='upper left', title=title)
        return

    def show(self):
        plt.show()
        return

if __name__ == "__main__":
    loss_img = data_show(('../logs/resnet18_dr__2022_06_17 22_07_04/train_loss_file.csv', '../logs/resnet18_dr__2022_06_17 22_07_04/val_loss_file.csv'))
    print(1)