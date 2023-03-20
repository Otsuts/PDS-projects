import torch
import numpy as np
import time
import argparse
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from dimreduce import DimReducer
from dataloader import AwA2Dataset

import random


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=int, default=2)
    parser.add_argument('--seed', type=int, default=114514)
    parser.add_argument('--reduce_method', type=str, default='LLE',
                        choices=['None', 'FeatureSelection', 'PCA', 'AutoEncoder', 'SNE', 'LLE'])
    parser.add_argument('--reduced_dim', type=int, default=64)

    return parser.parse_args()


def main(args):
    random.seed(args.seed)
    # 生成数据集
    dataset = AwA2Dataset()
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.4, train_size=0.6, random_state=args.seed)

    # 特征降维
    dim_reducer = DimReducer(train_dataset, test_dataset, method=args.reduce_method, dim=args.reduced_dim)
    start_time = time.time()
    X_train, y_train, X_test, y_test = dim_reducer.dim_reduce()
    print(f'Dimension reduction finished within time {time.time() - start_time:.4f}')

    # 构建SVM
    SVM = SVC(C=args.C, kernel='linear', decision_function_shape='ovr')
    # 开始训练SVM
    start_time = time.time()
    SVM.fit(X_train, y_train)
    print(f'Training finished within time {time.time() - start_time:.4f}')

    # 用训练好的SVM进行预测
    start_time = time.time()
    pred = SVM.predict(X_test)
    print(
        f'Prediction finished within time {time.time() - start_time:.4f} with precision {len(np.where(np.array(pred) == np.array(y_test))[0]) / test_dataset.__len__():.4f}')


if __name__ == '__main__':
    args = arg_parse()
    main(args)
