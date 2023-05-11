import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, args, in_dim=2048, hidden_dim=128, num_labels=85) -> None:
        super().__init__()
        self.args = args
        if not args.use_big:
            self.fc = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_labels),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_dim, 2000),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(2000, 1200),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(1200, 1200),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(1200, num_labels),
                nn.Sigmoid()
            )

    def forward(self, X):
        return self.fc(X)  # bs*dim


class Generator(nn.Module):
    def __init__(self, z_dim, attr_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim + attr_dim, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, x_dim, attr_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(x_dim + attr_dim, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class SemRelClassifier(nn.Module):
    def __init__(self, args, in_dim=2048, hidden_dim=128, output_dim=50) -> None:
        super().__init__()
        if args.use_big:
            self.fc = nn.Sequential(
                nn.Linear(in_dim, 2000),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(2000, 1200),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(1200, 1200),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(1200, output_dim),
                nn.Softmax(dim=1)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Softmax(dim=1)
            )

    def forward(self, X):
        return self.fc(X)  # bs*50


class MLPClassifier(nn.Module):
    def __init__(self, x_dim, attr_dim, out_dim):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(x_dim + attr_dim, 2000),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(2000, 1200),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(1200, 1200),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(1200, out_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)
