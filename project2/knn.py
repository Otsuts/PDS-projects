from sklearn.neighbors import KNeighborsClassifier
from metric_learn import NCA, MLKR, LFDA
from utils import write_log
from learner import Learner
import numpy as np
import time
import os


class KNN():
    def __init__(self, args):
        self.args = args
        self.method = args.method
        self.k = args.k_neighbors
        self.metric_learning = args.metric_learning
        self.methods_dict = {
            'Euclidean': 2,
            'Manhattan': 1
        }
        self.learner = Learner(args)
        self.classifier = KNeighborsClassifier(n_neighbors=self.k,
                                               p=self.methods_dict[self.method],
                                               )

    def fit(self, x_train, y_train):
        write_log(
            f'Using metric learning {self.metric_learning} method with {self.method} distance, n_components = {self.args.n_components} and k= {self.k} to fit',
            args=self.args)
        if not os.path.exists(f'data/{self.args.metric_learning}_{self.args.n_components}_train.npy'):
            start_time = time.time()
            x_train = self.learner.fit_transform(x_train, y_train)
            write_log(
                f'Learner transform using {time.time()-start_time:.4f}', self.args)
            np.save(
                f'data/{self.args.metric_learning}_{self.args.n_components}_train', x_train)
        else:
            write_log(
                f'load transformed from data/{self.args.metric_learning}_{self.args.n_components}_train.npy', self.args)
            x_train = np.load(
                f'data/{self.args.metric_learning}_{self.args.n_components}_train.npy')
        self.classifier.fit(x_train, y_train)

    def predict(self, x_test):
        if not os.path.exists(f'data/{self.args.metric_learning}_{self.args.n_components}_test.npy'):
            start_time = time.time()
            x_test = self.learner.transform(x_test)
            write_log(
                f'Learner transform using {time.time()-start_time:.4f}', self.args)
            np.save(
                f'data/{self.args.metric_learning}_{self.args.n_components}_test', x_test)
        else:
            x_test = np.load(
                f'data/{self.args.metric_learning}_{self.args.n_components}_test.npy')

        return self.classifier.predict(x_test)
