import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


# model = Sequential()
# model.add(Dense(64, activation='relu', input_dim=784))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(10, activation='softmax'))
# adam = Adam(lr=0.0001)
# model.compile(loss='categorical_crossentropy',  optimizer=adam,  metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=50, batch_size=256, verbose=2)


def validate(net, loader):
    net.eval()
    correct = 0
    counter = 0
    for (inputs, targets) in loader:
        inputs, targets = inputs.cuda(0), targets.cuda(0)
        y = net(inputs)
        y_pred = y.max(1)[1]
        correct += y_pred.eq(targets).sum()
        counter+=inputs.size(0)
    accuracy = correct / counter
    return accuracy.item()

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 64, bias=True)
        self.fc2 = nn.Linear(64, 256, bias=True)
        self.fc3 = nn.Linear(256, 10, bias=True)


    def forward(self, x):
        x = x.view(-1, 784) # 60000 x 28 x 28 -> 60000 x 784
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader =  torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

net = Net().cuda(0)

cross_entropy = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for epoch in range(10):
    net.train()
    loss_epoch = 0
    for (X_train, y_train) in train_loader:
        inputs = X_train.cuda(0)
        targets = y_train.cuda(0)

        optimizer.zero_grad()

        y = net(inputs)
        loss = cross_entropy(y, targets)
        loss_epoch+=loss.item()
        loss.backward()

        optimizer.step()

    print("Epoch ", epoch)
    print("Total loss/epoch = ", loss_epoch/len(train_loader))
    print("Train accuracy = ", validate(net, train_loader))
    print("Test accuracy = ", validate(net, test_loader))
