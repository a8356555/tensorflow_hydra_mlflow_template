import os

import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt


def save_confusion_matrix(cmatrix, classes: list, img_name: str, project_path, fig_title=None):
    cmatrix = pd.DataFrame(cmatrix, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=2.0)  # label size
    ax = sn.heatmap(cmatrix, annot=True, fmt='d')
    ax.set(xlabel='Predicted class', ylabel='True class')
    if fig_title is not None:
        plt.title(fig_title)
    plt.savefig(os.path.join(project_path, 'reports/figures', img_name + '.png'))
