import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle = True)


class AE(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder layers
        self.e1 = nn.Linear(784, 128)
        self.e2 = nn.Linear(128, 64)
        self.e3 = nn.Linear(64, 32)

        # Decoder layers
        self.d1 = nn.Linear(32, 64)
        self.d2 = nn.Linear(64, 128)
        self.d3 = nn.Linear(128, 784)

    def get_representation(self, x):
        x = self.e1(x)
        x = F.relu(x)
        x = self.e2(x)
        x = F.relu(x)
        x = self.e3(x)
        x = F.relu(x)
        return x

    def forward(self, x):
        x = self.e1(x)
        x = F.relu(x)
        x = self.e2(x)
        x = F.relu(x)
        x = self.e3(x)
        x = F.relu(x)

        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        x = F.relu(x)
        x = self.d3(x)
        x = F.sigmoid(x)
        return x


model = AE().cuda()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

from tqdm import tqdm

for epoch in tqdm(range(5)):
    loss_epoch = 0
    for (image, _) in train_loader:
        image = image.reshape(-1, 28*28).cuda(0)
        reconstructed = model(image)

        loss = loss_fn(reconstructed, image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch +=loss.item()
    print("Epoch ", epoch, " loss = ", loss_epoch)


train_representations = []
train_labels = []
for (image, labels) in train_loader:
    train_labels.append(labels)
    image = image.reshape(-1, 28 * 28).cuda(0)
    representation = model.get_representation(image)
    train_representations.append(representation.detach().cpu().numpy())

train_representations = np.concatenate(train_representations).reshape((-1, 32))
train_labels = np.concatenate(train_labels).reshape((-1, ))

test_representations = []
test_labels = []
for (image, labels) in test_loader:
    test_labels.append(labels)
    image = image.reshape(-1, 28 * 28).cuda(0)
    representation = model.get_representation(image)
    test_representations.append(representation.detach().cpu().numpy())

test_representations = np.concatenate(test_representations).reshape((-1, 32))
test_labels = np.concatenate(test_labels).reshape((-1, ))

print(train_representations.shape, train_labels.shape)
print(test_representations.shape, test_labels.shape)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

svc = KNeighborsClassifier(n_neighbors=5)
svc.fit(train_representations, train_labels)
print(svc.score(train_representations, train_labels))
print(svc.score(test_representations, test_labels))