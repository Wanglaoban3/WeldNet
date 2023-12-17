from torch.utils.data import Dataset
import json
import os
import cv2

class my_dataset(Dataset):
    def __init__(self, json_path, transform):
        self.root_path = os.path.abspath(os.path.join(json_path, os.pardir))
        self.transform = transform
        with open(json_path) as f:
            self.data = json.load(f)
        self.imgs = []
        self.labels = []
        for img, label in self.data.items():
            self.imgs.append(img)
            self.labels.append(label)
        print('成功加载数据集')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_path, self.imgs[index])
        # print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # img = img.convert('RGB')
        # if img.mode != "RGB":
        #     raise ValueError("Img is not RGB type!")
        if self.transform != None:
            transformed_image = self.transform(image=img)['image']
        label = self.labels[index]
        return transformed_image, label


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import albumentations as A
    import albumentations.pytorch.transforms as T
    import numpy as np

    # batch_size = 1
    # root_path = 'C:/wrd/al5083/al5083'
    # def collate_fn(batch):
    #     imgs = [data[0] for data in batch]
    #     ori_imgs = [data[1] for data in batch]
    #     return imgs, ori_imgs
    # class test_dataset(my_dataset):
    #     def __init__(self, json_path, transform):
    #         super(test_dataset, self).__init__(json_path, transform)
    #     def __getitem__(self, item):
    #         img_path = os.path.join(self.root_path, self.imgs[item])
    #         ori_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #         img = ori_img
    #         if self.transform != None:
    #             transformed_image = self.transform(image=img)['image']
    #         return transformed_image, ori_img
    #
    #
    # train_transform = A.Compose([
    #     A.RandomCrop(width=650, height=750),  # 随机裁剪
    #     A.HorizontalFlip(p=0.5),  # 水平翻转
    #     A.Rotate(limit=30),  # 随机旋转（正负45度）
    #     A.RandomBrightnessContrast(),  # 随机亮度和对比度调整
    #     A.GaussianBlur(),  # 高斯模糊
    #     A.GaussNoise(var_limit=(10.0, 50.0)),  # 随机高斯噪声
    #     A.Resize(width=800, height=974),  # 放缩
    #     A.ToFloat(max_value=255),
    # ])
    #
    # train_ann_path = root_path + '/train/train.json'
    # train_dataset = test_dataset(train_ann_path, transform=train_transform)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    #
    # for index, (imgs, ori_imgs) in enumerate(train_loader):
    #     img = imgs[0] * 255
    #     ori_img = ori_imgs[0]
    #     result = np.concatenate((img, ori_img), axis=1)
    #     # 保存拼接后的图片
    #     cv2.imwrite(os.path.join('E:/test_logs/al5803/data_augmention_imgs', str(index)+'.jpg'), result)
    #     if index > 100:
    #         break

    img_path = "C:/wrd/al5083/al5083/train/170906-114912-Al 2mm/frame_00225.png"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    flip = A.HorizontalFlip(p=1)
    img1 = flip(image=img)['image']
    randomCrop = A.RandomCrop(width=650, height=731)
    img2 = randomCrop(image=img)['image']
    resize = A.Resize(width=800, height=974)
    img2 = resize(image=img2)['image']

    rotate = A.Rotate(30, p=1)
    img3 = rotate(image=img)['image']
    bright = A.RandomBrightnessContrast(p=1)
    img4 = bright(image=img)['image']

    # cv2.imwrite('D:/wrd/my_papers/WeldNet/HorizontalFlip.png', img1)
    cv2.imwrite('D:/wrd/my_papers/WeldNet/RandomCrop.png', img2)
    # cv2.imwrite('D:/wrd/my_papers/WeldNet/Rotate.png', img3)
    # cv2.imwrite('D:/wrd/my_papers/WeldNet/Bright.png', img4)

