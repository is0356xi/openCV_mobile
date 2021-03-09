import torchvision.transforms as transforms
### 画像の前処理を定義
data_transforms = {
    "train" : transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.RandomResizedCrop(224),                       # --> ノイズ低減
                                  transforms.RandomHorizontalFlip(),                            # --> データに幅を持たせる・特徴量増やす
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])     # --> 特徴量を増やす・NNからすると正規化されてた方が特徴を見つけやすい(値の差に関係なく特徴を見つけやすい)
    ]),
    "val" : transforms.Compose([
                                  transforms.Resize(256,256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.458, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 正規化しない前処理
to_tensor_transforms = transforms.Compose([
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor()
])



import torchvision.datasets as datasets
### データセットの作成
data_folder = "./train_images"
transform = data_transforms["train"]

data = datasets.ImageFolder(root=data_folder, transform=transform)


import torch
### trainとtestの分離
train_ratio = 0.8
train_size = int(train_ratio * len(data))

val_size = len(data) - train_size
data_size = {"train":train_size, "val":val_size}
data_train, data_val = torch.utils.data.random_split(data, [train_size, val_size])




### データロータを作成
batch_size = 2
train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False)

dataloaders = {"train":train_loader, "val":test_loader}


import torchvision
import matplotlib.pyplot as plt
import numpy as np
### 画像の確認
def imshow(img):
    img = img / 2 + 0.5     
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 訓練データをランダムに取得
dataiter = iter(dataloaders["train"])
images, labels = dataiter.next()
print(labels)

# 画像の表示
imshow(torchvision.utils.make_grid(images))
# ラベルの表示
print(' '.join('%5s' % labels[labels[j]] for j in range(2)))