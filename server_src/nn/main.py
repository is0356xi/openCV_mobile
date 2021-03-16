import torchvision.transforms as transforms
### 画像の前処理を定義
data_transforms = {
    "train" : transforms.Compose([
                                  transforms.RandomResizedCrop(224),                       # --> ノイズ低減
                                  transforms.RandomHorizontalFlip(),                            # --> データに幅を持たせる・特徴量増やす
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])     # --> 特徴量を増やす・NNからすると正規化されてた方が特徴を見つけやすい(値の差に関係なく特徴を見つけやすい)
    ]),
    "val" : transforms.Compose([
                                  transforms.Resize(256),
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


import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset, TensorDataset, random_split
from torchvision import transforms

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# データ読み込み  
data_folder = "./random_split_data"
dataset = datasets.ImageFolder(root=data_folder)

# trainとtestの分離
train_ratio = 0.65

train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size

# lengths = [int(len(dataset)*train_ratio), int(len(dataset)*(1-train_ratio))]

data_size = {"train":train_size, "val":val_size}
dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size, val_size])
# dataset_train, dataset_val = torch.utils.data.random_split(dataset, lengths)

# データセットを作成
data_train = DatasetFromSubset(dataset_train, data_transforms["train"])
data_val = DatasetFromSubset(dataset_val, data_transforms["val"])


# データローダを作成
batch_size = 5
train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)

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

print(len(dataloaders["train"].dataset))
print(len(dataloaders["val"].dataset))

# 訓練データをランダムに取得
dataiter = iter(dataloaders["train"])
images, labels = dataiter.next()
print("size = {}".format(images.size()))
print("labels = {}".format(labels))

# 画像の表示
imshow(torchvision.utils.make_grid(images))
# ラベルの表示
# print(' '.join('%5s' % labels[labels[j]] for j in range(5)))
print(' '.join('%5s' % labels[j] for j in range(5)))


import torch.nn as nn
### サイズチェック
dataiter = iter(dataloaders["train"])
images, labels = dataiter.next()

features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

print(features(images).size())
print(features(images).view(images.size(0), -1).size())
fc_size = features(images).view(images.size(0), -1).size()[1]
print(fc_size)



### ネットワーク・損失関数・最適化関数の設定
import torch.nn as nn
import torch.optim as optim

class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(fc_size, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

device = "cuda"
net = AlexNet().to(device)
# net = AlexNet()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)


### 学習 ###
num_epochs = 100

# プロット用のリスト
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0

    # trainモード
    net.train()
    for i, (images, labels) in enumerate(dataloaders["train"]):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        train_acc += (outputs.max(1)[1] == labels).sum().item()
        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / len(dataloaders["train"].dataset)
    avg_train_acc = train_acc / len(dataloaders["train"].dataset)

    # valモード
    net.eval()
    with torch.no_grad():
        for images, labels in dataloaders["val"]:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += (outputs.max(1)[1] == labels).sum().item()

    avg_val_loss = val_loss / len(dataloaders["val"].dataset)
    avg_val_acc = val_acc / len(dataloaders["val"].dataset)

    # trainデータ/valデータのloss, accuracy表示
    print('Epoch [{}/{}] , Loss: {loss:.4f}, val_Loss: {val_loss:.4f}, val_acc: {val_acc:.4f}:'.format(epoch+1, num_epochs, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))

    # グラフへプロットするためにリストへ格納
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)