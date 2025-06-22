

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt

# Set environment variables for reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.manual_seed(40)
torch.cuda.manual_seed_all(40)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Installing the dataset
# !!!!! Make Sure you run this on Colab cuz these commands work there or Change the commands !!!!

# UnComment these two lines if you are using Colab else write your own commands.

'''
!wget -O Caltech101_dataset.zip "https://www.kaggle.com/api/v1/datasets/download/imbikramsaha/caltech-101"
!unzip Caltech101_dataset.zip -d Caltech101_data
'''

# Remodelling the dataset according to our plan
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])
full_dataset = torchvision.datasets.ImageFolder(root='./Caltech101_data/caltech-101',transform=transform)
class_names = full_dataset.classes

train_size = int(0.7 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_data, val_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])


# Creating batches of size 32
batch_size = 32
train_batch = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
val_batch = torch.utils.data.DataLoader(val_data,batch_size=batch_size,shuffle=True)
test_batch = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=True)

# Function to train our model
def train_model(model, train_data, test_data, val_data, num_epochs, loss_func, optimizer, device):
  train_loss = []
  val_acc = []
  for i in range(num_epochs):
    model.train()
    tot_loss = 0
    for data in train_data:
      X, y = data
      X, y = X.to(device), y.to(device)
      optimizer.zero_grad()
      y_pred = model(X)
      loss = loss_func(y_pred, y)
      loss.backward()
      optimizer.step()
      tot_loss+=loss
    avg_loss = (tot_loss/len(train_data))
    print(f"Epoch: {i+1}, Avg Loss: {avg_loss:.4f}")
    train_loss.append(avg_loss)
    model.eval()
    with torch.no_grad():
      correct = 0
      total = 0
      for data in val_data:
        X, y = data
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        total += y.size(0)
        correct += (y_pred.argmax(1)==y).sum().item()
      acc = (correct/total)*100
      print(f"Accuracy: {acc:.2f}")
      val_acc.append(acc)

  print("Training done.")
  model.eval()
  with torch.no_grad():
      correct = 0
      total = 0
      for data in test_data:
        X, y = data
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        total += y.size(0)
        correct += (y_pred.argmax(1)==y).sum().item()
      print(f"Accuracy of Trained Model on Testing Data: {correct/total}")
  return train_loss,val_acc

# Defining our model
# This is the best and stable model I created for Caltech101 dataset

model = nn.Sequential(nn.BatchNorm2d(3),
                      nn.Conv2d(3,64,7,stride=2,padding=3,bias=False),nn.BatchNorm2d(64),nn.LeakyReLU(negative_slope=0.05),nn.MaxPool2d(2,2), #56,56
                      nn.Conv2d(64,128,5,padding=2),nn.BatchNorm2d(128),nn.LeakyReLU(negative_slope=0.05),nn.MaxPool2d(2,2), #28,28
                      nn.Conv2d(128,256,5,padding=2),nn.BatchNorm2d(256),nn.LeakyReLU(negative_slope=0.05),nn.MaxPool2d(2,2), #14,14
                      nn.Conv2d(256,512,5),nn.BatchNorm2d(512),nn.LeakyReLU(negative_slope=0.05),nn.MaxPool2d(2,2), #5,5
                      nn.Flatten(),
                      nn.Linear(512*5*5,4096),nn.BatchNorm1d(4096),nn.LeakyReLU(negative_slope=0.05),
                      nn.Dropout(p=0.3),
                      nn.Linear(4096,1024),nn.BatchNorm1d(1024),nn.LeakyReLU(negative_slope=0.05),
                      nn.Dropout(p=0.2),
                      nn.Linear(1024,256),nn.BatchNorm1d(256),nn.LeakyReLU(negative_slope=0.05),
                      nn.Dropout(p=0.2),
                      nn.Linear(256,102))

# Training our model
model.to(device)
learning_rate = 0.01
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 40 #Change the number of epochs according to your processor.
train_loss,val_acc = train_model(model, train_batch, test_batch, val_batch,num_epochs, loss_func, optimizer, device)

#Use the lists and plot the Training Loss and Validation Accuracies.


'''
I am Providing some more models I experimented, Do check them out.

model = nn.Sequential(nn.BatchNorm2d(3),                     
                      nn.Conv2d(3,64,7,stride=2,padding=3,bias=False),nn.BatchNorm2d(64),nn.LeakyReLU(negative_slope=0.05),nn.MaxPool2d(2,2), #56,56
                      nn.Conv2d(64,128,5,padding=2),nn.BatchNorm2d(128),nn.LeakyReLU(negative_slope=0.05),nn.MaxPool2d(2,2), #28,28
                      nn.Conv2d(128,256,5,padding=2),nn.BatchNorm2d(256),nn.LeakyReLU(negative_slope=0.05),nn.MaxPool2d(2,2), #14,14
                      nn.Conv2d(256,512,5),nn.BatchNorm2d(512),nn.LeakyReLU(negative_slope=0.05),nn.MaxPool2d(2,2), #5,5
                      nn.Flatten(),
                      nn.Linear(512*5*5,4096),nn.BatchNorm1d(4096),nn.LeakyReLU(negative_slope=0.05),
                      nn.Dropout(p=0.3),
                      nn.Linear(4096,1024),nn.BatchNorm1d(1024),nn.LeakyReLU(negative_slope=0.05),
                      nn.Dropout(p=0.2),
                      nn.Linear(1024,256),nn.BatchNorm1d(256),nn.LeakyReLU(negative_slope=0.05),
                      nn.Dropout(p=0.2),
                      nn.Linear(256,128),nn.BatchNorm1d(128),nn.LeakyReLU(negative_slope=0.05),
                      nn.Linear(128,102))

model = nn.Sequential(nn.BatchNorm2d(3),           
                      nn.Conv2d(3,64,7,stride=2,padding=3,bias=False),nn.BatchNorm2d(64),nn.LeakyReLU(negative_slope=0.05),nn.MaxPool2d(2,2), #56,56
                      nn.Conv2d(64,128,5,padding=2),nn.BatchNorm2d(128),nn.LeakyReLU(negative_slope=0.05),nn.MaxPool2d(2,2), #28,28
                      nn.Conv2d(128,256,5,padding=2),nn.BatchNorm2d(256),nn.LeakyReLU(negative_slope=0.05),nn.MaxPool2d(2,2), #14,14
                      nn.Conv2d(256,512,5),nn.BatchNorm2d(512),nn.LeakyReLU(negative_slope=0.05),nn.MaxPool2d(2,2), #5,5
                      nn.Flatten(),
                      nn.Linear(512*5*5,4096),nn.BatchNorm1d(4096),nn.LeakyReLU(negative_slope=0.05),
                      nn.Dropout(p=0.3),
                      nn.Linear(4096,1024),nn.BatchNorm1d(1024),nn.LeakyReLU(negative_slope=0.05),
                      nn.Dropout(p=0.2),
                      nn.Linear(1024,512),nn.BatchNorm1d(512),nn.LeakyReLU(negative_slope=0.05),
                      nn.Dropout(p=0.8),
                      nn.Linear(512,256),nn.BatchNorm1d(256),nn.LeakyReLU(negative_slope=0.05),
                      nn.Dropout(p=0.2),
                      nn.Linear(256,102))

model = nn.Sequential(nn.Conv2d(3,64,7,stride=2,padding=3,bias=False),nn.BatchNorm2d(64),nn.LeakyReLU(negative_slope=0.05),nn.MaxPool2d(2,2), #56,56
                      nn.Conv2d(64,128,5,padding=2),nn.BatchNorm2d(128),nn.LeakyReLU(negative_slope=0.05),nn.MaxPool2d(2,2), #28,28
                      nn.Conv2d(128,256,5,padding=2),nn.BatchNorm2d(256),nn.LeakyReLU(negative_slope=0.05),nn.AvgPool2d(2,2), #14,14
                      nn.Conv2d(256,512,5),nn.BatchNorm2d(512),nn.LeakyReLU(negative_slope=0.05),nn.AvgPool2d(2,2), #5,5
                      nn.Flatten(),
                      nn.Linear(512*5*5,4096),nn.BatchNorm1d(4096),nn.LeakyReLU(negative_slope=0.05),
                      nn.Linear(4096,1024),nn.BatchNorm1d(1024),nn.LeakyReLU(negative_slope=0.05),
                      nn.Linear(1024,512),nn.BatchNorm1d(512),nn.LeakyReLU(negative_slope=0.05),
                      nn.Linear(512,256),nn.BatchNorm1d(256),nn.LeakyReLU(negative_slope=0.05),
                      nn.Linear(256,102))
'''