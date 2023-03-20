import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import AutoEncoder
from sklearn.feature_selection import SelectKBest, chi2
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA


class DimReducer():
    def __init__(self, train_dataset, test_dataset, method='None',dim=64):
        self.train_loader = DataLoader(train_dataset, batch_size=train_dataset.__len__())
        self.test_loader = DataLoader(test_dataset, batch_size=test_dataset.__len__())
        self.method = method
        self.reduced_dim = dim
        self.dim_reduce_func = {
            'None': self.reduce_none,
            'FeatureSelection': self.reduce_feature_selection,
            'PCA': self.reduce_PCA,
            'AutoEncoder': self.reduce_autoencoder,
            'SNE':self.reduce_SNE,
            'LLE':self.reduce_LLE

        }

    def dim_reduce(self):
        return self.dim_reduce_func[self.method]()

    def reduce_none(self):
        for train_data, train_label in self.train_loader:
            X_train = train_data.numpy()
            y_train = train_label.numpy()

        for test_data, test_label in self.test_loader:
            X_test = test_data.numpy()
            y_test = test_label.numpy()

        print(f'Reduced dimension: {X_train.shape}')
        return X_train, y_train, X_test, y_test

    def reduce_SNE(self):
        for train_data, train_label in self.train_loader:
            X_train = train_data.numpy()
            y_train = train_label.numpy()

        for test_data, test_label in self.test_loader:
            X_test = test_data.numpy()
            y_test = test_label.numpy()

        tsne = TSNE(n_components=3,learning_rate=100)
        X_new_train = tsne.fit_transform(X_train,y_train)
        X_new_test = tsne.fit(X_test)
        print(f'Reduced dimension: {X_new_train.shape}')
        return X_new_train, y_train, X_new_test, y_test

    def reduce_LLE(self):
        for train_data, train_label in self.train_loader:
            X_train = train_data.numpy()
            y_train = train_label.numpy()

        for test_data, test_label in self.test_loader:
            X_test = test_data.numpy()
            y_test = test_label.numpy()


        lle = LocallyLinearEmbedding(n_neighbors=30,n_components=self.reduced_dim,method='standard')
        X_new_train = lle.fit_transform(X_train)
        X_new_test = lle.transform(X_test)
        print(f'Reduced dimension: {X_train.shape}')
        return X_new_train, y_train, X_new_test, y_test

    def reduce_PCA(self):
        for train_data, train_label in self.train_loader:
            X_train = train_data.numpy()
            y_train = train_label.numpy()

        for test_data, test_label in self.test_loader:
            X_test = test_data.numpy()
            y_test = test_label.numpy()
        pca = PCA(n_components=self.reduced_dim)
        X_new_train = pca.fit_transform(X_train, y_train)
        X_new_test = pca.transform(X_test)
        print(f'Reduced dimension: {X_train.shape}')
        return X_new_train, y_train, X_new_test, y_test


    def reduce_feature_selection(self):
        for train_data, train_label in self.train_loader:
            X_train = train_data.numpy()
            y_train = train_label.numpy()

        for test_data, test_label in self.test_loader:
            X_test = test_data.numpy()
            y_test = test_label.numpy()
        selector = SelectKBest(chi2, k=self.reduced_dim)
        X_new_train = selector.fit_transform(X_train, y_train)
        X_new_test = selector.transform(X_test)
        print(f'Reduced dimension: {X_new_train.shape}')
        return X_new_train, y_train, X_new_test, y_test

    def reduce_autoencoder(self):
        # train_autoencoder
        AE = AutoEncoder(2048,hidden_dim=self.reduced_dim)
        AE.to(AE.device)
        optimizer = torch.optim.Adam(AE.parameters(), lr=1e-4)
        loss_func = nn.MSELoss()
        print('start training AutoEncoder')
        for epoch in tqdm(range(250)):
            loss_all = 0.0
            step = 0
            for X_train, _ in self.train_loader:
                X_train = X_train.to(AE.device)
                target = X_train
                encoded, decoded = AE(X_train)
                loss = loss_func(decoded, target)

                optimizer.zero_grad()
                loss.backward()
                loss_all += loss
                optimizer.step()
                step += 1

        AE.cpu()
        # dimension reduction
        for train_data, train_label in self.train_loader:
            X_train = AE(train_data)[0].detach().numpy()
            y_train = train_label.numpy()

        for test_data, test_label in self.test_loader:
            X_test = AE(test_data)[0].detach().numpy()
            y_test = test_label.numpy()
        print(f'Reduced dimension: {X_train.shape}')
        return X_train, y_train, X_test, y_test
