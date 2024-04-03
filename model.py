from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import albumentations
import albumentations.pytorch
# Function fopr Training the model of the train dataset.
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.input = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # CONVOLUTION BLOCK 1
        # Depthwise sperable layer
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=30, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=3, padding=1, groups=3), #### Depthwise convolution
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.Conv2d(in_channels=30, out_channels=32, kernel_size=1), ### Pointwise Convolution (using depthwise output) <-- This is called DEPTHWISE-SEPARABLE CONVOLUTION
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # CONVOLUTION BLOCK 2 - Dialated Convolution
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False,dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # CONVOLUTION BLOCK 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # CONVOLUTION BLOCK 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        )

        self.dropout = nn.Dropout(0.01)

        # First fully connected layer
        self.fc1 = nn.Linear(128*8*8, 100) # [ [BATCH_SIZE] * [CHANNEL_NUMBER] * [HEIGHT] * [WIDTH].]
        self.fc2 = nn.Linear(100, 10) # [ [BATCH_SIZE] * [CHANNEL_NUMBER] * [HEIGHT] * [WIDTH].]

    def forward(self, x):
        x = self.input(x)
        #print('input layer - ',x.shape)
        x = self.convblock1(x)
        #print('conv1 - ',x.shape)
        x = self.convblock2(x)
        #print('conv2 - ',x.shape)
        x = self.convblock3(x)
        #print('conv3 - ',x.shape)
        x = self.convblock4(x)
        #print('conv4 - ',x.shape)
        x = self.gap(x)
        #print('Gap - ',x.shape)
        #print(x.view(-1, x.shape[-1]))
        x = x.view(-1, 128*8*8) #[CHANNEL_NUMBER] * [HEIGHT] * [WIDTH]
        #print('Flatten - ', x.view(-1, x.shape[-1]).shape[0])
        x = self.fc1(x)
        x = self.fc2(x)
        #print('Linear - ',x.shape)
        return F.log_softmax(x, dim=-1)

class ResNet_Custom(nn.Module):
    def __init__(self):
        super(ResNet_Custom, self).__init__()
        # Input Block
        self.input = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )


        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.resblock1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # CONVOLUTION BLOCK 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=0, bias=False),
            nn.MaxPool2d(kernel_size = 3, stride = 3),
            nn.BatchNorm2d(256),
            nn.ReLU()

        )

        # CONVOLUTION BLOCK 3
        self.convblock3 = nn.Sequential(
            # Main Block
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # Res Block
        self.resblock3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
            #Add  both the block
        )

        # CONVOLUTION BLOCK 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(kernel_size = 4,stride = 4),
            nn.ReLU(inplace=True)


        )

        self.maxpool = nn.MaxPool2d(kernel_size = 4,stride = 4)

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        )

        self.dropout = nn.Dropout(0.01)

        # First fully connected layer
        self.fc1 = nn.Linear(512*1*1, 10) # [ [BATCH_SIZE] * [CHANNEL_NUMBER] * [HEIGHT] * [WIDTH].]

    def forward(self, x):
        x = self.input(x)
        #print('input',x.shape)
        x = self.convblock1(x)
        #print('COnv1',x.shape)
        resblock = self.resblock1(x)
        #print('resblock',resblock.shape)
        x=x+resblock
       #print('First Add',x.shape)
        x = self.convblock2(x)
        #print('conv2',x.shape)

        x = self.convblock3(x)
        #print('conv3',x.shape)
        resblock3 = self.resblock3(x.clone())
        #print('Second Resblock',resblock3.shape)
        x=x+resblock3
        #print('Second Add',x.shape)

        #x = self.convblock4(x)
        x = self.maxpool(x)
        #print('Conv4',x.shape)
        #x = self.gap(x)
        #print('GAP',x.shape)
        x = x.view(-1, 512*1*1) #[CHANNEL_NUMBER] * [HEIGHT] * [WIDTH]
        #print('View',x.shape)
        #print(x.view(-1, x.shape[-1]).shape[0])
        x = self.fc1(x)
        #print('FC',x.shape)
        return F.log_softmax(x, dim=-1)



class Train_Module:
    def __init__(self,model,device,criterion,optimizer,train_loader,test_loader,num_epochs,batch_size):
        super(Train_Module,self).__init__()
        self.model =model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer 
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.epochs = num_epochs
       
    
    def GetCorrectPredCount(pPrediction, pLabels):
      return pPrediction.argmax(dim=1).eq(pLabels).sum().item()
    
    
    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        pbar = tqdm(train_loader)
    
        train_loss = 0
        correct = 0
        processed = 0
    
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
    
            # Predict
            output = model(data)
    
            # Calculate loss
            loss = F.nll_loss(output, target)
            train_loss+=loss.item()
    
            # Backpropagation
            loss.backward(retain_graph=True)
            optimizer.step()
    
            correct += GetCorrectPredCount(output, target)
            processed += len(data)
            #pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')
            pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    
    
        train_acc.append(100*correct/processed)
        train_losses.append(train_loss/len(train_loader))
        #return(train_acc,train_losses)
    
    
    # Testing the trained model on test dataset to the check loss and model accuracy
    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
    
        test_loss /= len(test_loader.dataset)
        test_acc.append(100. * correct / len(test_loader.dataset))
        test_losses.append(test_loss)
    
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        #return(test_acc,test_losses)

    def __call__(self):
        for epoch in range(self.epochs):
            print("EPOCH:", epoch+1)
            self.train(self.model, self.device, self.train_loader, self.optimizer, self.epochs)
        
            self.test(self.model, self.device, self.test_loader)


import torch
import torch.nn.functional as F
from tqdm import tqdm

class ModelTrainer:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

    def get_correct_pred_count(self, prediction, labels):
        return prediction.argmax(dim=1).eq(labels).sum().item()

    def train(self, model, device, train_loader, optimizer, epoch,criterion):
        model.train()
        pbar = tqdm(train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Predict
            output = model(data)

            # Calculate loss
            #loss = F.nll_loss(output, target)
            loss = criterion(output, target)
            train_loss += loss.item()

            # Backpropagation
            loss.backward(retain_graph=True)
            optimizer.step()

            correct += self.get_correct_pred_count(output, target)
            processed += len(data)
            pbar.set_description(desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

        self.train_acc.append(100*correct/processed)
        self.train_losses.append(train_loss/len(train_loader))

    def test(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        self.test_acc.append(100. * correct / len(test_loader.dataset))
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    def display_plot(self):
      fig, axs = plt.subplots(2,2,figsize=(15,10))
      axs[0, 0].plot(self.train_losses)
      axs[0, 0].set_title("Training Loss")
      axs[1, 0].plot(self.train_acc)
      axs[1, 0].set_title("Training Accuracy")
      axs[0, 1].plot(self.test_losses)
      axs[0, 1].set_title("Test Loss")
      axs[1, 1].plot(self.test_acc)
      axs[1, 1].set_title("Test Accuracy")

