import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, num_classes,bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.fc4(x)
        x = self.relu(x)
        x_bn4 = self.bn4(x)
        x = self.fc5(x_bn4)
        return x, x_bn4


torch.manual_seed(0)
excel_file_path = 'feature3-v2.xlsx'
df = pd.read_excel(excel_file_path)
selected_columns = df.iloc[:, 3:24]
numpy_array = selected_columns.to_numpy()
x = numpy_array[:, 0:7]
y = numpy_array[:, 20].astype(int)-1
mask = ~np.isnan(x).any(axis=1)  # 创建一个掩码，其中非NaN行为True
x = x[mask]
y = y[mask]

features = torch.tensor(x, dtype=torch.float32)
labels = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(features, labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = features.shape[1]
num_classes = len(set(labels.numpy()))
model = SimpleNN(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model = model.cuda()
criterion = criterion.cuda()

num_epochs = 100
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
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        outputs, _ = model(data)
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
            outputs, _ = model(data)
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
plt.savefig('results3-v2/nn_classification_loss_curve.png')
plt.close()

# Plot Accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.legend()
plt.savefig('results3-v2/nn_classification_accuracy_curve.png')
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
        outputs, _ = model(data)
        predicted= torch.argmax(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

report = classification_report(all_targets, all_preds, digits=4)
print(report)

# Save classification report to Excel
report_dict = classification_report(all_targets, all_preds, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_excel('results3-v2/nn_classification_report.xlsx', index=True)

# Plot and save confusion matrix
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('results3-v2/nn_confusion_matrix.png')
plt.close()
