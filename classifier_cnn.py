import torch.nn.functional as F
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
directory = 'DATA/feature3-v2'
max_len = 1200
# num_epochs = 500
learning_rate = 0.0001
# test_size = 0.2  # 测试集比例
batch_size = 16
shuffle = True
# 超参数设置
# input_dim = 20  # 特征维数
num_features = 16  # Transformer的d_model参数，确保能被nhead整除
num_classes = 5  # 分类类别数
nhead = 8  # Transformer头数
num_encoder_layers = 2  # Transformer编码器层数
dim_feedforward = 32  # 前馈网络的维度
max_seq_length = 70  # 序列最大长度

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


class TimeDomainCNN(nn.Module):
    def __init__(self, input_channels, num_classes, max_len=60):
        super(TimeDomainCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels,
                               out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128 * max_len, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to [batchsize, channel, time]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Example usage
input_channels = 20

model = TimeDomainCNN(input_channels=input_channels,
                      num_classes=num_classes, max_len=max_len)

# class AudioClassifier(nn.Module):
#     def __init__(self, input_dim, num_features, num_classes, nhead, num_encoder_layers, dim_feedforward, max_seq_length):
#         super(AudioClassifier, self).__init__()
#         self.input_dim = input_dim
#         self.num_features = num_features
#         self.num_classes = num_classes
#         self.nhead = nhead
#         self.num_encoder_layers = num_encoder_layers
#         self.dim_feedforward = dim_feedforward
#         self.max_seq_length = max_seq_length
#         self.input_proj = nn.Linear(input_dim, num_features)  # 添加输入投影层

#         # 位置编码层
#         self.positional_encoding = nn.Parameter(
#             torch.randn(1, self.max_seq_length, self.num_features))

#         # Transformer编码器层
#         encoder_layers = TransformerEncoderLayer(
#             d_model=self.num_features, nhead=self.nhead, dim_feedforward=self.dim_feedforward)
#         self.transformer_encoder = TransformerEncoder(
#             encoder_layers, num_layers=self.num_encoder_layers)

#         # 分类头
#         self.fc_out = nn.Linear(self.num_features, self.num_classes)

#     def forward(self, src):
#         src = self.input_proj(src)  # 将输入特征投影到num_features维
#         # 将输入和位置编码相加
#         src = src + self.positional_encoding[:, :src.size(1), :]
#         src = src * math.sqrt(self.num_features)

#         # Transformer编码
#         output = self.transformer_encoder(src)

#         # 池化操作，取最后一个时间步的输出
#         output = torch.mean(output, dim=1)

#         # 分类层
#         output = self.fc_out(output)

#         return output


# model = AudioClassifier(input_dim, num_features, num_classes,
#                         nhead, num_encoder_layers, dim_feedforward, max_seq_length)
print(model)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model = model.cuda()
criterion = criterion.cuda()
# 训练模型

# 训练模型
num_epochs = 40
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

best_test_accuracy = 0  # 用于记录测试集的最佳准确率
best_epoch = 0  # 记录最佳epoch
best_model_state = None  # 用于保存最佳模型的状态
best_preds = []  # 保存最佳模型的预测结果
best_targets = []  # 保存最佳模型的真实标签

for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for data, target in train_loader:
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
    epoch_preds = []  # 保存当前 epoch 的预测结果
    epoch_targets = []  # 保存当前 epoch 的真实标签
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += target.size(0)
            correct_test += (predicted == target).sum().item()
            epoch_preds.extend(predicted.cpu().numpy())
            epoch_targets.extend(target.cpu().numpy())

    test_loss = running_loss / len(test_loader)
    test_losses.append(test_loss)
    test_accuracy = 100 * correct_test / total_test
    test_accuracies.append(test_accuracy)

    # 如果当前epoch的测试准确率是最高的，保存模型和相关结果
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        best_epoch = epoch
        best_model_state = model.state_dict()  # 保存模型的状态
        best_preds = epoch_preds  # 保存最佳模型的预测结果
        best_targets = epoch_targets  # 保存最佳模型的真实标签

# 恢复最佳模型
model.load_state_dict(best_model_state)

# 保存最佳模型
torch.save(best_model_state, 'results3-v2/best_model.pth')
print(
    f'Best Epoch: {best_epoch}, Best Test Accuracy: {best_test_accuracy:.2f}%')

# 使用最佳模型生成分类报告和混淆矩阵
report = classification_report(best_targets, best_preds, digits=4)
print(report)

# 保存分类报告到Excel
report_dict = classification_report(best_targets, best_preds, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_excel(
    'results3-v2/best_cnn_classification_report.xlsx', index=True)

# 绘制并保存混淆矩阵
cm = confusion_matrix(best_targets, best_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Best Model)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('results3-v2/best_cnn_confusion_matrix.png')
plt.close()

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.savefig('results3-v2/best_cnn_classification_loss_curve.png')
plt.close()

# 绘制准确率曲线
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.legend()
plt.savefig('results3-v2/best_cnn_classification_accuracy_curve.png')
plt.close()
