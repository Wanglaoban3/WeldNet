import json
import os
import random

random.seed(12)
probability = 80
json_files = ['铝合金数据集/al5083/test/test.json', '铝合金数据集/al5083/train/train.json']
total_data = {}
imgs = []
labels = []
for i in json_files:
    root_path = os.path.split(i)[0].split('铝合金数据集/')[1] + '/'
    with open('../' + i) as f:
        data = json.load(f)
    for img, label in data.items():
        imgs.append(root_path+img)
        labels.append(label)

train_resplit = {}
test_resplit = {}


data_len = len(imgs)
for index, i in enumerate(imgs):
    rand = random.randint(0, 100)
    if rand > probability:
        train_resplit.update({i: labels[index]})
    else:
        test_resplit.update({i: labels[index]})

# train_resplit = json.dumps(train_resplit)
# test_resplit = json.dumps(test_resplit)

with open('../铝合金数据集/al5083/train/train_resplit.json', 'w', encoding='utf-8') as json_file:
    json.dump(train_resplit, json_file)
with open('../铝合金数据集/al5083/test/test_resplit.json', 'w', encoding='utf-8') as json_file:
    json.dump(test_resplit, json_file)

# 把json文件换行写入的办法
# data4 = json.dumps(data3, indent=1)
# with open('c:wrd/铝合金数据集/al5083/semi-supervised_train.json', 'w', newline='\n') as json_file:
#     json_file.write(data4)