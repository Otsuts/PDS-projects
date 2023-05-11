import torch
import os
import numpy as np
from torch.utils.data import Dataset


class BiasDataset(Dataset):
    def __init__(self, dataset_name='testclasses.txt', data_path='../Animals_with_Attributes2/Features/ResNet101'):
        super().__init__()
        self.predicate_binary_mat = np.array(np.genfromtxt(
            os.path.join(data_path, 'predicate-matrix-binary.txt'), dtype='int'))
        # get class and label dict
        self.class_to_index = {}
        self.index_to_class = {}
        with open(os.path.join(data_path, 'classes.txt')) as f:
            index = 0
            for line in f:
                class_name = line.split('\t')[1].strip()
                self.class_to_index[class_name] = index
                self.index_to_class[index] = class_name
                index += 1
        # get class names in wanted dataset
        self.label_available = []
        self.classes_names = set()
        with open(os.path.join(data_path, 'AWA2_class_split', dataset_name)) as f:
            for line in f:
                self.classes_names.add(line.split('\n')[0].strip())
                self.label_available.append(
                    self.class_to_index[line.split('\n')[0].strip()])
        # construct dataset
        self.label_available = list(set(self.label_available))
        data = []
        labels = []
        # label from zero
        with open(os.path.join(data_path, 'AwA2-features.txt'), 'r')as feature, open(os.path.join(data_path, 'AwA2-labels.txt')) as label:
            feature_list = feature.readlines()
            label_list = label.readlines()
            for index, label in enumerate(label_list):
                if self.index_to_class[int(label) - 1] in self.classes_names:
                    labels.append(int(label)-1)
                    data.append(
                        list(map(float, feature_list[index].strip().split())))

        self.data = torch.tensor(data, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.int)
        print(f'Data loaded with shape {self.data.shape}')

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, index):
        label_index = self.labels[index]
        return self.data[index, :], torch.tensor(self.predicate_binary_mat[label_index, :], dtype=torch.int), label_index


class SynDataset(Dataset):
    def __init__(self, syndataset) -> None:
        super().__init__()
        data = []
        feature = []
        label = []
        for d, f, l in syndataset:
            data.append(d.tolist())
            feature.append(f.tolist())
            label.append(l)
        self.data = torch.tensor(data, dtype=torch.float)
        self.labels = torch.tensor(label, dtype=torch.int)
        self.feature = torch.tensor(feature, dtype=torch.float)

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, index):
        return self.data[index, :], self.feature[index, :], self.labels[index]


if __name__ == '__main__':
    Dataset = BiasDataset()
    print(Dataset.__getitem__([0, 1]))
