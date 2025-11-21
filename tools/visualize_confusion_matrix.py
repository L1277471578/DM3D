import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from utils import parser
from tqdm import tqdm
from utils import parser
import os
matplotlib.use('Agg')

args1 = parser.get_args()
path = args1.model_path + '/fig'
if not os.path.exists(path):
    os.mkdir(path)


# 定义一个 4x4 的混淆矩阵
def confusion_matrix2(config,cm_percent,class_names=None):
    # cm_percent = np.array([
    #     [0.98, 0.0, 0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0052, 0],
    #     [0.0, 0.99, 0.00, 0.0052, 0.00, 0.0, 0, 0, 0, 0],
    #     [0, 0.0, 1, 0, 0.00, 0, 0.0, 0.0, 0, 0],
    #     [0.00, 0.02, 0.0, 0.98, 0.0, 0, 0.00, 0, 0.0, 0],
    #     [0.0047, 0.0, 0.00, 0.0, 0.99,  0.00, 0, 0.0,0.0047, 0.00],
    #     [0.0, 0.0, 0.0, 0.00, 0.0, 1, 0.00, 0.0, 0.00, 0.0],
    #     [0.015, 0, 0.00, 0.0, 0.0049, 0, 1, 0, 0.00, 0.00],
    #     [0.0, 0, 0, 0.0, 0.0, 0.00, 0, 1, 0, 0.0],
    #     [0.00, 0.0, 0, 0.0, 0.00, 0.00, 0.00, 0, 1, 0],
    #     [0.00, 0.0, 0.0, 0, 0.0051, 0.0, 0.00, 0.0, 0, 0.99]
    # ])

    if config.model.cls_dim > 20:
        labelsize=4

    else:
        labelsize = 9

    # 总样本数
    total_samples = 1000

    # 将百分比矩阵转换为样本数矩阵
    cm = np.round(cm_percent * total_samples).astype(int)

    # 生成真实标签和预测标签
    true_labels = []
    pred_labels = []

    print('混淆矩阵图生成...........')
    for true_class in  tqdm(range(cm.shape[0]),mininterval=0.1):
        for pred_class in range(cm.shape[1]):
            count = cm[true_class, pred_class]
            true_labels.extend([true_class] * count)
            pred_labels.extend([pred_class] * count)

    # 创建一个 DataFrame
    data = {
        'True Label': true_labels,
        'Predicted Label': pred_labels
    }
    df = pd.DataFrame(data)
    # df.to_csv('Predicted.csv')

    cm = confusion_matrix(df["True Label"], df["Predicted Label"], labels=class_names)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.round(cm, 2)

    plt.figure(figsize=(9, 7))
    heatmap = sns.heatmap(cm, annot=True, cmap="Blues", annot_kws={"size": labelsize, "fontweight": "bold"}, cbar=True)
    plt.xlabel("Predicted", fontsize=(labelsize+5))
    plt.ylabel("True", fontsize=(labelsize+5))
    plt.xticks(size=labelsize)
    plt.yticks(size=labelsize)

    # Set the font size of the colorbar label
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=(labelsize+1))


    plt.savefig(args1.model_path+'/混淆矩阵.jpg', dpi=1200, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(path + "/混淆矩阵.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(path + "/混淆矩阵.eps",dpi=1200, bbox_inches='tight', pad_inches=0.1)
