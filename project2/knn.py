from sklearn.neighbors import KNeighborsClassifier
from metric_learn import NCA,MLKR,LFDA
from utils import write_log


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
        self.learner = None
        self.classifier = KNeighborsClassifier(n_neighbors=self.k,
                                               p=self.methods_dict[self.method],
                                               )

    def fit(self, x_train, y_train):
        write_log(
            f'Using metric learning {self.metric_learning} method with {self.method} distance and k= {self.k} to fit',
            args=self.args)
        if self.metric_learning == 'NCA':
            self.learner = NCA(max_iter=1000)
            x_train = self.learner.fit_transform(x_train, y_train)
        if self.metric_learning == 'MLKR':
            self.learner = MLKR()
            x_train = self.learner.fit_transform(x_train,y_train)
        if self.metric_learning == 'LFDA':
            self.learner = LFDA(k=self.k)
            x_train = self.learner.fit_transform(x_train,y_train)
        self.classifier.fit(x_train, y_train)

    def predict(self, x_test):
        if self.learner:
            self.learner.transform(x_test)
        return self.classifier.predict(x_test)
