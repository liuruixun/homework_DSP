import os
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import re
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def calculate_dtw(signal1, signal2):
    distance, path = fastdtw(signal1, signal2, dist=euclidean)
    return distance


test_path = 'features/test'
folder_path = 'features/train'

# 获取测试集和训练集中的文件数量
test_files = [f for f in os.listdir(test_path) if f.endswith('.npy')]
train_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

# 初始化预测和真实标签数组
num_pre = np.zeros((len(test_files)))
num_gt = np.zeros((len(test_files)))

# 存储每个测试文件的最小DTW距离和对应的模板文件名
min_distances = []
predicted_labels = []
num=100
# 遍历测试集文件
for i, filename_test in enumerate(tqdm(test_files)):
    file_path_test = os.path.join(test_path, filename_test)
    data_test = np.load(file_path_test)[:, 7:21]
    numbers_test = re.findall(r'\d+', filename_test)
    num_gt[i] = int(numbers_test[0])

    # 初始化最小DTW距离和对应的模板文件名
    min_distance = float('inf')
    closest_template = None

    # 遍历训练集文件
    for filename_train in tqdm(train_files):
        file_path_train = os.path.join(folder_path, filename_train)
        data_train = np.load(file_path_train)[:, 7:21]
        distance = calculate_dtw(data_test, data_train)

        # 更新最小DTW距离和对应的模板文件名
        if distance < min_distance:
            min_distance = distance
            closest_template = filename_train

    # 存储最小DTW距离和对应的模板文件名
    min_distances.append(min_distance)
    predicted_label = int(re.findall(r'\d+', closest_template)[0])
    predicted_labels.append(predicted_label)

    # 更新预测数组
    num_pre[i] = predicted_label
    if i==num:
        break

# 计算性能指标
accuracy = accuracy_score(num_gt[:num], predicted_labels[:num])
precision = precision_score(
    num_gt[:num], predicted_labels[:num], average='weighted')
recall = recall_score(num_gt[:num], predicted_labels[:num], average='weighted')
f1 = f1_score(num_gt[:num], predicted_labels[:num], average='weighted')

# 打印性能指标
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
