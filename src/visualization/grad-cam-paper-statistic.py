import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df = pd.read_csv('grad-cam.csv')
    df['label_binary'] = df['label'].apply(lambda x: 1 if x == 4 else 0)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(df.columns)

    # sns.barplot(x='label_binary', y='mean', data=test_df, hue='label_binary', capsize=0.1, dodge=False)
    # plt.savefig('reports/grad-cam/grad-cam.png')
    # plt.show()

    output = test_df[['label_binary', 'important_list']]

    colors = ['b', 'r']

    for label, important_list in output.values:
        important_list = np.array(eval(important_list))
        important_list_plot = np.mean(important_list, axis=(1, 2))
        plt.plot(important_list_plot, color=colors[label], alpha=0.1)

    plt.title('Train')
    plt.savefig('reports/grad-cam/grad-cam-frame.png')
