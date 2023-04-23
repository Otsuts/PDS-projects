import torch
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from utils import write_log


class AwA2Dataset(Dataset):
    def __init__(self, args, file_path='./Animals_with_Attributes2/Features/ResNet101', ):
        super().__init__()
        self.args = args
        data_path = os.path.join(file_path, 'AwA2-features.txt')
        label_path = os.path.join(file_path, 'AwA2-labels.txt')
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(list(map(float, line.strip().split())))

        self.data = torch.tensor(data, dtype=torch.float)
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                labels.append(int(line) - 1)
        self.labels = torch.tensor(labels, dtype=torch.int)
        write_log(f'Data loaded with shape {self.data.shape}', args)

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, index):
        return self.data[index, :], self.labels[index]


# testing
if __name__ == '__main__':
    pass
