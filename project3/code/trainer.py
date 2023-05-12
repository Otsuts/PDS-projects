import torch
import sys
import torch.nn as nn
import numpy as np
from datasets import BiasDataset
from torch.utils.data import DataLoader, ConcatDataset
from torch.distributions import normal
from utils import *
from models import Classifier, Generator, Discriminator, MLPClassifier, SemRelClassifier
from datasets import SynDataset
from torch.autograd import Variable


class SemRel():
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = SemRelClassifier(args).to(self.device)
        self.train_dataset = BiasDataset(dataset_name='trainvalclasses.txt')
        self.test_dataset = BiasDataset(dataset_name='testclasses.txt')
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=args.batch_size, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.learning_rate)
        self.similarity_matrix = self.get_similarity_matrix(
            predicate_map=self.train_dataset.predicate_binary_mat)

    def train(self):
        for epoch in range(self.args.num_epochs):
            loss = 0
            acc = 0
            for data, feature, label in self.train_loader:
                l, a = self.fit_classifier(data, feature, label)
                loss += l
                acc += a
            write_log(
                f'Epoch[{epoch}] pretrain loss {loss/self.train_loader.__len__():.4f} training acc {acc/self.train_loader.__len__():.4f}', self.args)
            # test on testing dataset
            if (epoch+1) % self.args.eval_iter == 0:
                curr_acc = self.evaluate(self.test_loader, self.test_dataset)
                write_log(
                    f'Epoch [{epoch+1}] with testing accuracy: {curr_acc:.4f}', self.args)

    def evaluate(self, dataloader, dataset):
        self.model.eval()
        mean_acc = 0.0
        pred_class = []
        true_class = []
        with torch.no_grad():
            for data, feature, label in dataloader:
                data = data.to(self.device)
                feature = feature.to(self.device).float()
                output = self.model(data)
                output = torch.matmul(
                    output, self.similarity_matrix.to(self.device)).softmax(dim=0)

                curr_pred_classes = self.label_to_class(
                    output, dataset.label_available)
                pred_class.extend(curr_pred_classes)

                curr_true_classes = []
                for index in label:
                    curr_true_classes.append(index.item())
                true_class.extend(curr_true_classes)
        pred_class = np.array(pred_class)
        true_class = np.array(true_class)
        mean_acc = np.mean(pred_class == true_class)
        return mean_acc

    def get_similarity_matrix(self, predicate_map):
        mat_size = predicate_map.shape[0]
        similarity_mat = torch.ones([mat_size, mat_size])
        for i in range(0, mat_size):
            for j in range(i+1, mat_size):
                similarity_mat[i][j] = similarity_mat[j][i] = torch.cosine_similarity(
                    torch.from_numpy(predicate_map[i, :]).float().unsqueeze(0), torch.from_numpy(predicate_map[j, :]).float().unsqueeze(0))
        return torch.nn.functional.normalize(similarity_mat)

    def fit_classifier(self, data, feature, label):
        data = data.to(self.device)
        label = label.to(self.device).long()
        y_pred = self.model(data)
        pred_label = self.label_to_class(
            y_pred, self.train_dataset.label_available)
        self.optimizer.zero_grad()
        loss = self.criterion(y_pred, label)
        loss.backward()
        self.optimizer.step()
        acc = np.mean(np.array(label.cpu()) == np.array(pred_label))
        return loss.item(), acc

    def label_to_class(self, pred_labels, label_available):
        predictions = []
        pred_labels_cp = pred_labels.clone().detach()
        pred_labels_cp[:, label_available] = 0
        pred = pred_labels - pred_labels_cp
        predictions = torch.argmax(pred, dim=1)
        return predictions.cpu()


class SemEmb():
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device(
            'cuda')if torch.cuda.is_available() else torch.device('cpu')
        self.model = Classifier(args).to(self.device)
        self.train_dataset = BiasDataset(dataset_name='trainvalclasses.txt')
        self.test_dataset = BiasDataset(dataset_name='testclasses.txt')
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=args.batch_size, shuffle=False)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.learning_rate)

    def train(self):
        for epoch in range(self.args.num_epochs):
            # scan through training data
            train_pred_class = []
            train_true_class = []
            for data, feature, label in self.train_loader:
                data = data.to(self.device)
                feature = feature.to(self.device).float()
                # train the model
                self.model.train()

                outputs = self.model(data)
                loss = self.criterion(outputs, feature)
                train_class = self.label_to_class(
                    outputs, self.train_dataset.label_available, self.train_dataset.predicate_binary_mat)
                train_pred_class.extend(train_class)
                curr_true_classes = []
                for index in label:
                    curr_true_classes.append(index.item())
                train_true_class.extend(curr_true_classes)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # evaluate on training dataset
            train_true_class = np.array(train_true_class)
            train_pred_class = np.array(train_pred_class)
            train_acc = np.mean(train_pred_class == train_true_class)
            write_log(
                f'Epoch [{epoch+1}] with training accuracy: {train_acc:.4f}', self.args)
            # test on test data
            if (epoch+1) % self.args.eval_iter == 0:
                curr_acc = self.evaluate(self.test_loader, self.test_dataset)
                write_log(
                    f'Epoch [{epoch+1}] with testing accuracy: {curr_acc:.4f}', self.args)

    def evaluate(self, dataloader, dataset):
        self.model.eval()
        mean_acc = 0.0
        pred_class = []
        true_class = []
        with torch.no_grad():
            for data, feature, label in dataloader:
                data = data.to(self.device)
                feature = feature.to(self.device).float()
                output = self.model(data)
                curr_pred_classes = self.label_to_class(
                    output, dataset.label_available, dataset.predicate_binary_mat)
                pred_class.extend(curr_pred_classes)

                curr_true_classes = []
                for index in label:
                    curr_true_classes.append(index.item())
                true_class.extend(curr_true_classes)
        pred_class = np.array(pred_class)
        true_class = np.array(true_class)
        mean_acc = np.mean(pred_class == true_class)
        return mean_acc

    def label_to_class(self, pred_labels, label_available, predicate_binary_mat):
        predictions = []
        for i in range(pred_labels.shape[0]):
            curr_labels = pred_labels[i, :].cpu().detach().numpy()
            best_dist = sys.maxsize
            best_index = -1
            for j in label_available:
                class_labels = predicate_binary_mat[j, :]
                dist = get_euclidean_dist(curr_labels, class_labels)
                if dist < best_dist:
                    best_index = j
                    best_dist = dist
            predictions.append(best_index)
        return predictions


class SynTrainer():
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        # define dataset
        self.train_dataset = BiasDataset(dataset_name='trainvalclasses.txt')
        self.test_dataset = BiasDataset(dataset_name='testclasses.txt')
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=args.batch_size, shuffle=False)
        # define latent variable z
        self.z_dim = args.z_dim
        self.z_distribution = normal.Normal(0, 1)
        self.z_shape = torch.Size([self.args.batch_size, self.z_dim])
        # define generator
        self.Generator = Generator(self.z_dim, 85).to(self.device)
        self.optim_G = torch.optim.Adam(
            self.Generator.parameters(), lr=self.args.g_lr)
        # define discriminator
        self.Discriminator = Discriminator(2048, 85).to(self.device)
        self.optim_D = torch.optim.Adam(
            self.Discriminator.parameters(), lr=args.d_lr)

        # define classifier for judging the output of generator
        self.classifier = MLPClassifier(2048, 85, 50).to(self.device)
        self.optim_cls = torch.optim.Adam(
            self.classifier.parameters(), lr=args.cls_lr)

        # define final test classifier
        self.final_classifier = MLPClassifier(2048, 85, 50).to(self.device)
        self.optim_final_cls = torch.optim.Adam(
            self.final_classifier.parameters(), lr=args.fcls_lr)

        self.criterion_cls = nn.CrossEntropyLoss()

    def get_conditional_input(self, X, C_Y):
        new_X = torch.cat([X, C_Y], dim=1).float()
        return Variable(new_X).to(self.device)

    def fit_classifier(self, data, feature, label):
        new_data = self.get_conditional_input(data, feature)
        label = label.to(self.device).long()
        y_pred = self.classifier(new_data)
        pred_label = self.label_to_class(
            y_pred, self.train_dataset.label_available)
        self.optim_cls.zero_grad()
        loss = self.criterion_cls(y_pred, label)
        loss.backward()
        self.optim_cls.step()
        acc = np.mean(np.array(label.cpu()) == np.array(pred_label))
        return loss.item(), acc

    def fit_GAN(self, data, feature, label, use_cls_loss=True):
        data = data.to(self.device)
        feature = feature.to(self.device)
        label = label.to(self.device)
        total_disc_loss = []
        # optimize discriminator for several times
        for _ in range(self.args.n_discriminator):
            X_real = self.get_conditional_input(data, feature)

            Z = self.z_distribution.sample(
                [feature.shape[0], self.z_dim]).to(self.device)
            Z = self.get_conditional_input(Z, feature)
            X_GEN = self.Generator(Z)
            X_GEN = self.get_conditional_input(X_GEN, feature)
            fake_label = torch.zeros(X_GEN.shape[0]).to(self.device)
            true_label = torch.ones(X_real.shape[0]).to(self.device)

            L_disc = (nn.BCELoss()(self.Discriminator(X_GEN).squeeze(), fake_label) +
                      nn.BCELoss()(self.Discriminator(X_real).squeeze(), true_label))/2

            total_disc_loss.append(L_disc.item())

            self.optim_D.zero_grad()
            L_disc.backward()
            self.optim_D.step()
        # optimize generator
        Z = self.z_distribution.sample(
            [feature.shape[0], self.z_dim]).to(self.device)
        Z = self.get_conditional_input(Z, feature)

        X_GEN = self.Generator(Z)

        X = torch.cat([X_GEN, feature], dim=1).to(self.device)
        L_gen = -torch.mean(self.Discriminator(X))
        if use_cls_loss:
            self.classifier.eval()
            y_pred = self.classifier(X)
            log_prob = torch.log(torch.gather(
                y_pred, 1, label.unsqueeze(1).long()))
            L_cls = -torch.mean(log_prob)
            L_gen += 0.01 * L_cls
        self.optim_G.zero_grad()
        L_gen.backward()
        self.optim_G.step()
        return L_gen, np.mean(np.array(total_disc_loss))

    def fit_final_classifier(self, data, feature, label):
        data = data.to(self.device)
        feature = feature.to(self.device)
        label = label.to(self.device)

        data = torch.cat([data, feature], dim=1).to(self.device)
        y_pred = self.final_classifier(data)

        self.optim_final_cls.zero_grad()
        loss = self.criterion_cls(y_pred, label)
        loss.backward()
        self.optim_final_cls.step()

        return loss.item()

    def create_syn_dataset(self, test_labels, predicate_binary_mat, n_samples=1000):
        syn_dataset = []
        for label_index in test_labels:
            attr = predicate_binary_mat[label_index, :]
            z = self.z_distribution.sample(torch.Size([n_samples, self.z_dim]))
            c_y = torch.stack([torch.FloatTensor(attr)
                              for _ in range(n_samples)])
            z_inp = torch.cat([z, c_y], dim=1).to(self.device)
            X_gen = self.Generator(z_inp)
            syn_dataset.extend([(X_gen[i], attr, label_index)
                               for i in range(n_samples)])
        return syn_dataset

    def train(self):
        # pretrain the classifier
        if os.path.exists(f'../models/pretrain_{self.args.model}_{self.args.learning_rate}.pth'):
            self.classifier.load_state_dict(torch.load(
                f'../models/pretrain_{self.args.model}_{self.args.learning_rate}.pth'))
            write_log('Pretrain model loaded', args=self.args)
        else:
            write_log('Pretraining classifier on training set', self.args)
            best_pretrain_acc = 0.0
            for epoch in range(self.args.pretrain_epochs):
                loss = 0
                acc = 0
                for data, feature, label in self.train_loader:
                    l, a = self.fit_classifier(data, feature, label)
                    loss += l
                    acc += a
                if acc > best_pretrain_acc:
                    torch.save(self.classifier.state_dict(
                    ), f'../models/pretrain_{self.args.model}_{self.args.learning_rate}.pth')
                write_log(
                    f'Epoch[{epoch}] pretrain loss {loss/self.train_loader.__len__():.4f} training acc {acc/self.train_loader.__len__():.4f}', self.args)
        # train gan
        if os.path.exists(f'../models/generator_{self.args.model}_{self.args.g_lr}_{self.args.gan_epochs}.pth') and \
                os.path.exists(f'../models/discriminator_{self.args.model}_{self.args.d_lr}_{self.args.gan_epochs}.pth'):
            self.Generator.load_state_dict(torch.load(
                f'../models/generator_{self.args.model}_{self.args.g_lr}_{self.args.gan_epochs}.pth'))
            self.Discriminator.load_state_dict(torch.load(
                f'../models/discriminator_{self.args.model}_{self.args.d_lr}_{self.args.gan_epochs}.pth'))
            write_log('GAN model loaded', self.args)
        else:
            write_log('Training gan', self.args)
            for epoch in range(self.args.gan_epochs):
                loss_disc = 0
                loss_gen = 0
                for data, feature, label in self.train_loader:
                    l_gen, l_disc = self.fit_GAN(data, feature, label, True)
                    loss_disc += l_disc
                    loss_gen += l_gen
                write_log(
                    f'Epoch[{epoch}] disc loss: {loss_disc:.4f}, gen loss: {loss_gen:.4f}', self.args)
            torch.save(self.Generator.state_dict(
            ), f'../models/generator_{self.args.model}_{self.args.g_lr}_{self.args.gan_epochs}.pth')
            torch.save(self.Discriminator.state_dict(
            ), f'../models/discriminator_{self.args.model}_{self.args.d_lr}_{self.args.gan_epochs}.pth')

        # train final classifier
        syn_dataset = SynDataset(self.create_syn_dataset(
            self.test_dataset.label_available, self.train_dataset.predicate_binary_mat, n_samples=self.args.num_samples))
        final_dataset = ConcatDataset([self.train_dataset, syn_dataset])
        final_train_loader = DataLoader(
            final_dataset, batch_size=self.args.batch_size, shuffle=True)
        best_test_acc = 0.0
        for epoch in range(self.args.num_epochs):
            train_pred_class = []
            train_true_class = []
            for data, feature, label in final_train_loader:
                data = self.get_conditional_input(data, feature)

                # train the model
                self.final_classifier.train()

                outputs = self.final_classifier(data)
                loss = self.criterion_cls(
                    outputs, label.to(self.device).long())
                train_class = self.label_to_class(
                    outputs, self.train_dataset.label_available,)
                train_pred_class.extend(train_class)
                curr_true_classes = []
                for index in label:
                    curr_true_classes.append(index.item())
                train_true_class.extend(curr_true_classes)

                self.optim_final_cls.zero_grad()
                loss.backward()
                self.optim_final_cls.step()
            # evaluate on training dataset
            train_true_class = np.array(train_true_class)
            train_pred_class = np.array(train_pred_class)
            train_acc = np.mean(train_pred_class == train_true_class)
            write_log(
                f'Epoch [{epoch+1}] with training accuracy: {train_acc:.4f}', self.args)
            # test on test data

            if (epoch+1) % self.args.eval_iter == 0:
                curr_acc = self.evaluate(self.test_loader, self.test_dataset)
                if curr_acc > best_test_acc:
                    best_test_acc = curr_acc
                write_log(
                    f'Epoch [{epoch+1}] with testing accuracy: {curr_acc:.4f}', self.args)
        write_log(f'Best acc :{best_test_acc:.4f}', self.args)

    def label_to_class(self, pred_labels, label_available):
        predictions = []
        pred_labels_cp = pred_labels.clone().detach()
        pred_labels_cp[:, label_available] = 0
        pred = pred_labels - pred_labels_cp
        predictions = torch.argmax(pred, dim=1)
        return predictions.cpu()

    def evaluate(self, dataloader, dataset):
        self.final_classifier.eval()
        mean_acc = 0.0
        pred_class = []
        true_class = []
        with torch.no_grad():
            for data, feature, label in dataloader:
                data = self.get_conditional_input(data, feature)
                output = self.final_classifier(data)
                curr_pred_classes = self.label_to_class(
                    output, dataset.label_available,)
                pred_class.extend(curr_pred_classes)

                curr_true_classes = []
                for index in label:
                    curr_true_classes.append(index.item())
                true_class.extend(curr_true_classes)
        pred_class = np.array(pred_class)
        true_class = np.array(true_class)
        mean_acc = np.mean(pred_class == true_class)
        return mean_acc
