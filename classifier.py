import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 指定Excel文件路径
excel_file_path = 'feature3-v2.xlsx'

# 读取Excel文件
df = pd.read_excel(excel_file_path)

# 选择第2至第11列（注意：Python索引从0开始，所以列号为1到10）
selected_columns = df.iloc[:, 3:24]

# 将选定的列转换为NumPy数组
numpy_array = selected_columns.to_numpy()
print(numpy_array.shape)
x = numpy_array[:, 0:20]
y = numpy_array[:, 20].astype(int)-1

# 删除x中含有NaN的行以及对应y中的行
mask = ~np.isnan(x).any(axis=1)  # 创建一个掩码，其中非NaN行为True
x = x[mask]
y = y[mask]

# 分割数据集为训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# 定义分类器列表
classifiers = {
    "SVM": SVC(kernel='linear'),
    "LDA": LinearDiscriminantAnalysis(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
}

# 创建一个空的DataFrame用于存储分类报告
report_df = pd.DataFrame()

# 测试每个分类器并打印分类报告
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 打印分类报告
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"Classification report for {name}:")
    print(classification_report(y_test, y_pred))
    print("\n")

    # 添加分类报告到DataFrame
    temp_df = pd.DataFrame(report).transpose()
    temp_df['classifier'] = name

    report_df = pd.concat([report_df, temp_df])

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'results3/{name}_confusion_matrix.png')
    plt.close()

# 保存分类报告到Excel
report_df.to_excel('results3/classification_reports.xlsx', index=False)
