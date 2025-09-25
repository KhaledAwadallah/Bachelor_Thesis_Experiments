import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.ensemble import RandomForestClassifier


class MoleculeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx, 2]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


class BaselineFHNN(nn.Module):
    def __init__(self, input_size, hs1, hs2, hs3, output_size):
        super(BaselineFHNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hs1)
        self.fc2 = nn.Linear(hs1, hs2)
        self.fc3 = nn.Linear(hs2, hs3)
        self.fc4 = nn.Linear(hs3, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.input_dropout = nn.Dropout(0.25)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.input_dropout(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.sigmoid(self.fc4(x))
        return x


class FineTunedFHNN(nn.Module):
    def __init__(self, input_size, hs1, output_size):
        super(FineTunedFHNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hs1)
        self.fc2 = nn.Linear(hs1, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.input_dropout = nn.Dropout(0.25)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.input_dropout(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x


class MAMLNN(nn.Module):
    def __init__(self, input_size, hs1, output_size):
        super(MAMLNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hs1)
        self.fc2 = nn.Linear(hs1, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.input_dropout = nn.Dropout(0.25)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.input_dropout(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

    def functional_forward(self, x, weights):
        x = nn.functional.linear(x, weights["fc1.weight"], weights["fc1.bias"])
        x = self.relu(x)
        x = nn.functional.linear(x, weights["fc2.weight"], weights["fc2.bias"])
        x = self.sigmoid(x)
        return x


def get_random_forest_classifier(seed):
    return RandomForestClassifier(random_state=seed)