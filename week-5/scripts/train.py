import argparse
import torch
import torch.nn as nn
import torchvision
from pokemon_classifier import CNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
The hyperparams that are set using argparse
lr = 0.01(def)
epochs = 5(def)
batch_size = 4(def)

We can also include the optimizer to set using the argparse, but no time to do that. I'll try to do that next time.
'''

parser =argparse.ArgumentParser()
parser.add_argument('-l','--lr',type=float,default=0.01,help='It sets the learning rate for the model. {default = %(default)s}')
parser.add_argument('-e','--epochs',type=int,default=5,help='It sets the number of epochs for the model. {default = %(default)s}')
parser.add_argument('-b','--batch_size',type=int,default=4,help='It sets the batch size for the model. {default = %(default)s}')
args = parser.parse_args()

lr = args.lr
epochs = args.epochs
batch_size = args.batch_size

'''
The above script gets the hyperparams from CLI. Please don't change it.
'''
model = CNN()
model.load_state_dict(torch.load("../models/trained_model-2.pth"))
model.to(device)

dataset = torchvision.datasets.ImageFolder(root='../data/processed')

train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_batch = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
val_batch = torch.utils.data.DataLoader(val_data, batch_size=batch_size, drop_last=True)
test_batch = torch.utils.data.DataLoader(test_data, batch_size=batch_size, drop_last=True)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for i in range(epochs):
    model.train()
    for (X, y) in train_batch:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_func(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f'Epoch: {i+1}/{epochs}, Loss: {loss.item()}')
    model.eval()
    with torch.no_grad():
        val_loss = 0
        correct = 0
        for X, y in val_batch:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            val_loss += loss_func(y_pred, y).item()
            correct += (y_pred.argmax(1)==y).sum().item()
        val_loss /= len(val_data)
        print(f'Validation Loss: {val_loss}, Accuracy: {100. * correct / len(val_data)}%')
print("Training complete.")

model.eval()
with torch.no_grad():
        test_loss = 0
        correct = 0
        for X, y in test_batch:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            test_loss += loss_func(y_pred, y).item()
            correct += (y_pred.argmax(1)==y).sum().item()
        test_loss /= len(test_data)
        print(f'Test Loss: {test_loss}, Accuracy: {100. * correct / len(test_data)}%')

torch.save(model.state_dict(), '../models/trained_model-2.pth')
