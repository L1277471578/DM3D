import os

import torch
import numpy as np
from tools import builder
from utils import misc, dist_utils, tsne_utils
from utils.logger import *
from sklearn.manifold import TSNE
# from openTSNE import TSNE
from openTSNE import TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization
from datasets import data_transforms
from torchvision import transforms

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from utils import parser
import matplotlib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tools.visualize_confusion_matrix import confusion_matrix2
# from tools.visualize_confusion_matrix import confusion_matrix2
args1 = parser.get_args()
path = args1.model_path + '/fig'
if not os.path.exists(path):
    os.mkdir(path)

test_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudScaleAndTranslate(),
    ]
)


def tsne_net(args, config):
    print('T-Sne start ... ')

    # build dataset
    (_, test_dataloader) = builder.dataset_builder(args, config.dataset.test)

    # build model
    base_model = builder.model_builder(config.model)
    # load ckpts
    if args.ckpts is not None:
        # builder.load_model(base_model, args.ckpts)
        base_model.load_model_from_ckpt(args.ckpts)  # for BERT
    else:
        print('Training from scratch')

    base_model = base_model.cuda()

    tsne(base_model, test_dataloader, args, config)


# visualization
def tsne(base_model, test_dataloader, args, config):
    base_model.eval()

    test_feat = []
    test_label = []
    npoints = config.npoints
    category_nums = config.model.cls_dim

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()
            points = misc.fps(points, npoints)

            feat = base_model(points)
            test_feat.append(feat.detach())

            target = label.view(-1)
            test_label.append(target.detach())

        test_feat = torch.cat(test_feat, dim=0)
        test_label = torch.cat(test_label, dim=0)
        
        test_feat = test_feat.cpu().numpy()
        test_label = test_label.cpu().numpy()

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(test_feat.squeeze())

    plot_embedding(result, test_label, category_nums, config)



def plot_embedding(data, label, category_nums, config):

    colors = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#42d4f4', '#f032e6', '#fabed4', '#469990',
              '#dcbeff', '#9A6324', '#800000', '#000075', '#a9a9a9', '#888870', '#000000'
              ]
    if category_nums > 27:
        base = [0, 0.3, 0.6, 0.9]
    else:
        base = [0, 0.5, 0.9]
    for i in range(len(base)):
        for j in range(len(base)):
            for k in range(len(base)):
                colors.append([base[i], base[j], base[k], 1])

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    matplotlib.use('Agg')

    if config.model.cls_dim >20:
        plt.figure(figsize=(5.5, 5))
    else:
        plt.figure(figsize=(6, 5))
    unique_labels = np.unique(label)
    for i, ul in enumerate(unique_labels):
        plt.scatter(data[label == ul, 0], data[label == ul, 1],
                    color=colors[i % len(colors)],
                    label=f"{ul}")

    legend_patches = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i % len(colors)], markersize=8,
                             label=f"{ul}") for i, ul in enumerate(unique_labels)]

    if config.model.cls_dim > 20:
        plt.legend(handles=legend_patches, labelspacing=0.3, columnspacing=0.2, handletextpad=0.1, loc='upper left',
                   bbox_to_anchor=(1.01, 1), ncol=2)
    else:
        plt.legend(handles=legend_patches, labelspacing=0.3, columnspacing=0.2, handletextpad=0.1, loc='upper left',
                   bbox_to_anchor=(1.01, 1), ncol=1)

    plt.title("T-SNE Visualization")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.axis('on')
    plt.savefig(args1.model_path + "/tsne.jpg", dpi=1200, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(path + "/tsne.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(path + "/tsne.eps", dpi=1200, bbox_inches='tight', pad_inches=0.1)
    print('tsne图生成完成')


def visualize_confusion_matrix(test_pred, test_label):
    pred_np = test_pred.cpu().numpy()
    label_np = test_label.cpu().numpy()
    cm = confusion_matrix(label_np, pred_np)
    return cm
