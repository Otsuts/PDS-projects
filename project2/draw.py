import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from os import walk
import seaborn as sns


def get_precision(metric, distance, K, N):
    for _, _, log_list in walk('./logs'):
        for file_name in log_list:
            split = file_name.split('_')
            cur_distance = split[1]
            cur_k = split[2]
            cur_n = split[3]
            cur_metric = split[-1].split('.')[0]
            if cur_distance == distance and cur_metric == metric and cur_k == K and cur_n == N:
                with open(f'./logs/{file_name}', 'r') as f:
                    the_line = f.readlines()[-1]
                    if str.isdigit(str(the_line.split('.')[-1][:-1])):
                        return float(the_line.split(' ')[-1])


def draw_bar():

    sns.set(color_codes=True)
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

    # 柱高信息
    Y = [0.8945, 0.9110, 0.9220, 0.9074]
    Y1 = [0.8848, 0.9147, 0.9213, 0.9004]
    X = np.arange(len(Y))

    bar_width = 0.25
    tick_label = ['Plain', 'NCA', 'LFDA', 'MLKR']

    # 显示每个柱的具体高度
    for x, y in zip(X, Y):
        plt.text(x-0.15, y+0.005, '%.3f' % y, ha='center', va='bottom')

    for x, y1 in zip(X, Y1):
        plt.text(x+0.25, y1+0.005, '%.3f' % y1, ha='center', va='bottom')

    # 绘制柱状图
    plt.bar(X, Y, bar_width, align="center",
            color="red", label="Euclidean", alpha=0.5)
    plt.bar(X+bar_width, Y1, bar_width, color="purple", align="center",
            label="Manhattan", alpha=0.5)

    plt.xlabel("Metric leaning methods")
    plt.ylabel("Accuracy")
    plt.title('Best accuracy of two distances')

    plt.xticks(X+bar_width/2, tick_label)
    # 显示图例
    plt.legend()
    # plt.show()
    plt.savefig('results/bar.png', dpi=400)


def draw_plain_knn():
    x = range(2, 21, 1)
    manhattan_plain = []
    euclidean_plain = []
    for k in x:
        manhattan_plain.append(get_precision(
            'None', 'Manhattan', str(k), '32'))
        euclidean_plain.append(get_precision(
            'None', 'Euclidean', str(k), '32'))
    plt.plot(x, manhattan_plain, 'o-', label='Manhattan')
    plt.plot(x, euclidean_plain, 'o-', label='Euclidean')

    plt.title('Different distance in plain knn')
    plt.xlabel('#Neighbors')
    plt.ylabel('Accuracy')

    # 添加图例
    plt.legend()
    plt.savefig('results/plain_knn.png')


def draw_metric_learning(metric):
    plt.clf()
    x = range(2, 21, 1)
    manhattan_plain = [[], [], [], []]
    euclidean_plain = [[], [], [], []]
    for k in x:
        for i, N in enumerate(['32', '64', '128', '256']):
            manhattan_plain[i].append(get_precision(
                metric, 'Manhattan', str(k), N))
            euclidean_plain[i].append(get_precision(
                metric, 'Euclidean', str(k), N))
    for i, N in enumerate(['32', '64', '128', '256']):
        plt.plot(x, manhattan_plain[i], 'o-', label=f'Manhattan{N}')

    plt.title(f'{metric} method result with Manhattan')
    plt.xlabel('#Neighbors')
    plt.ylabel('Accuracy')

    # 添加图例
    plt.legend()
    plt.savefig(f'results/{metric}_knn_manhattan.png')

    plt.clf()
    for i, N in enumerate(['32', '64', '128', '256']):
        plt.plot(x, euclidean_plain[i], 'o-', label=f'Euclidean{N}')

    plt.title(f'{metric} method result with Euclidean')
    plt.xlabel('#Neighbors')
    plt.ylabel('Accuracy')

    # 添加图例
    plt.legend()
    plt.savefig(f'results/{metric}_knn_euclidean.png')


if __name__ == '__main__':
    # draw_plain_knn()
    # draw_metric_learning('MLKR')
    # draw_metric_learning('NCA')
    # draw_metric_learning('LFDA')
    draw_bar()
