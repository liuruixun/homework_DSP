import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import re
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
# 定义数据集类
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class AudioFeaturesDataset(Dataset):
    def __init__(self, directory, max_len=None, mode='train'):
        self.directory = os.path.join(directory, mode)
        # print(self.directory)
        self.max_len = max_len
        self.file_names = [f for f in os.listdir(
            self.directory) if os.path.isfile(os.path.join(self.directory, f))]
        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.file_names[idx])

        feature = np.load(file_path)
        file_name = self.file_names[idx]
        # feature = feature[:,:4]
        parts = file_name.split('_')
        numbers = re.findall(r'\d+', file_name)
        label = int(numbers[1])
        # print(len(feature))
        if self.max_len is not None:
            if len(feature) > self.max_len:
                feature = feature[:self.max_len]
            elif len(feature) < self.max_len:
                feature = np.pad(
                    feature, ((0, self.max_len - len(feature)), (0, 0)), 'constant', constant_values=0)

        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label-1, dtype=torch.int64)


# 设置参数
directory = 'DATA/feature3'
max_len = 1200
# num_epochs = 500
learning_rate = 0.001
# test_size = 0.2  # 测试集比例
batch_size = 64
shuffle = True
# 超参数设置
input_dim = 20  # 特征维数
num_features = 16  # Transformer的d_model参数，确保能被nhead整除
num_classes = 5  # 分类类别数
nhead = 8  # Transformer头数
num_encoder_layers = 2  # Transformer编码器层数
dim_feedforward = 32  # 前馈网络的维度
max_seq_length = 1200  # 序列最大长度

# 创建Dataset实例
train_dataset = AudioFeaturesDataset(
    directory=directory, max_len=max_len, mode='train')
test_dataset = AudioFeaturesDataset(
    directory=directory, max_len=max_len, mode='test')
# 划分训练集和测试集
# train_size = int((1 - test_size) * len(dataset))
# test_size = int(test_size * len(dataset))
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建DataLoader实例
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=shuffle)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型


class AudioClassifier(nn.Module):
    def __init__(self, input_dim, num_features, num_classes, nhead, num_encoder_layers, dim_feedforward, max_seq_length):
        super(AudioClassifier, self).__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        self.num_classes = num_classes
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.max_seq_length = max_seq_length
        self.input_proj = nn.Linear(input_dim, num_features)  # 添加输入投影层

        # 位置编码层
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.max_seq_length, self.num_features))

        # Transformer编码器层
        encoder_layers = TransformerEncoderLayer(
            d_model=self.num_features, nhead=self.nhead, dim_feedforward=self.dim_feedforward)
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers=self.num_encoder_layers)

        # 分类头
        self.fc_out = nn.Linear(self.num_features, self.num_classes)

    def forward(self, src):
        src = self.input_proj(src)  # 将输入特征投影到num_features维
        # 将输入和位置编码相加
        src = src + self.positional_encoding[:, :src.size(1), :]
        src = src * math.sqrt(self.num_features)

        # Transformer编码
        output = self.transformer_encoder(src)

        # 池化操作，取最后一个时间步的输出
        output = torch.mean(output, dim=1)

        # 分类层
        output = self.fc_out(output)

        return output


model = AudioClassifier(input_dim, num_features, num_classes,
                        nhead, num_encoder_layers, dim_feedforward, max_seq_length)
print(model)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model = model.cuda()
criterion = criterion.cuda()
# 训练模型

num_epochs = 20
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for data, target in train_loader:
        # print(data.shape)
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += target.size(0)
        correct_train += (predicted == target).sum().item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracies.append(100 * correct_train / total_train)

    model.eval()
    running_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += target.size(0)
            correct_test += (predicted == target).sum().item()

    test_loss = running_loss / len(test_loader)
    test_losses.append(test_loss)
    test_accuracies.append(100 * correct_test / total_test)

# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.savefig('results3-v2/transformer_classification_loss_curve.png')
plt.close()

# Plot Accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.legend()
plt.savefig('results3-v2/transformer_classification_accuracy_curve.png')
plt.close()

# # t-SNE Visualization
# model.eval()
# features_list = []
# labels_list = []
# with torch.no_grad():
#     for data, target in test_loader:
#         data = data.cuda()
#         _, features_bn4 = model(data)
#         features_list.append(features_bn4.cpu().numpy())
#         labels_list.append(target.numpy())

# features_array = np.concatenate(features_list, axis=0)
# labels_array = np.concatenate(labels_list, axis=0)

# # Select only the first five classes
# indices = np.isin(labels_array, [0, 1, 2])
# features_array = features_array[indices]
# labels_array = labels_array[indices]

# tsne = TSNE(n_components=2, random_state=0)
# features_tsne = tsne.fit_transform(features_array)

# plt.figure(figsize=(10, 8))
# sns.scatterplot(x=features_tsne[:, 0], y=features_tsne[:, 1],
#                 hue=labels_array, palette='viridis', legend='full')
# plt.title('t-SNE of Last Layer Features (First 5 Classes)')
# plt.savefig('tsne_visualization.png')
# plt.close()


all_preds = []
all_targets = []
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        outputs = model(data)
        predicted = torch.argmax(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

report = classification_report(all_targets, all_preds, digits=4)
print(report)

# Save classification report to Excel
report_dict = classification_report(all_targets, all_preds, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_excel(
    'results3-v2/transformer_classification_report.xlsx', index=True)

# Plot and save confusion matrix
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('results3-v2/transformer_confusion_matrix.png')
plt.close()
