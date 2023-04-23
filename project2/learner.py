from metric_learn import NCA, MLKR,LFDA


class Learner():
    def __init__(self, args):
        self.args = args

        if args.metric_learning == 'NCA':
            self.learner = NCA(init='pca', n_components=args.n_components, random_state=args.seed)
        elif args.metric_learning == 'MLKR':
            self.learner = MLKR(init='pca', n_components=args.n_components, random_state=args.seed)
        elif args.metric_learning == 'LFDA':
            self.learner = LFDA(n_components=args.n_components)
        else:
            self.learner = None

    def fit_transform(self, x_train, y_train):
        if self.learner is None:
            return x_train
        return self.learner.fit_transform(x_train, y_train)

    def transform(self, x_test):
        if self.learner is None:
            return x_test
        return self.learner.transform(x_test)
