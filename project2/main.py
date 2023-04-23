import argparse
import numpy as np
from knn import KNN
from utils import get_data
from sklearn.model_selection import train_test_split
from dataloader import AwA2Dataset
from utils import write_log, set_seed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='Manhattan', type=str)
    parser.add_argument('--n_components', default=64, type=int)
    parser.add_argument('--k_neighbors', default=5, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--log_path', default='./logs', type=str)
    parser.add_argument('--metric_learning', default='MLKR',
                        type=str, choices=['NCA', 'MLKR', 'LFDA', 'None'])

    return parser.parse_args()


def main(args):
    # generate dataset
    dataset = AwA2Dataset(args)
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.4, train_size=0.6, random_state=args.seed)
    x_train, y_train, x_test, y_test = get_data(train_dataset, test_dataset)

    # define classifier based on specified method
    clf = KNN(args)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    write_log(
        f'Training finishes with precision {len(np.where(np.array(pred) == np.array(y_test))[0]) / test_dataset.__len__():.4f}',
        args)


if __name__ == '__main__':
    set_seed()
    args = get_args()
    main(args)
