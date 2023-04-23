import datetime
import os
from torch.utils.data import DataLoader
import random
import torch
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_data(train_dataset, test_dataset):
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=train_dataset.__len__(), shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=test_dataset.__len__(), shuffle=False)
    x_train, y_train, x_test, y_test = None, None, None, None
    for x, y in train_loader:
        x_train = x.numpy()
        y_train = y.numpy()

    for x, y in test_loader:
        x_test = x.numpy()
        y_test = y.numpy()
    return x_train, y_train, x_test, y_test


def write_log(w, args):
    file_name = args.log_path + '/' + datetime.date.today().strftime('%m%d') + \
        f"_{args.method}_{args.k_neighbors}_{args.n_components}_{args.metric_learning}.log"
    if not os.path.exists(args.log_path + '/'):
        os.mkdir(args.log_path + '/')
    t0 = datetime.datetime.now().strftime('%H:%M:%S')
    info = "{} : {}".format(t0, w)
    print(info)
    with open(file_name, 'a') as f:
        f.write(info + '\n')
