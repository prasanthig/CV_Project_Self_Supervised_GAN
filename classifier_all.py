
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
from model_new import Discriminator
from dataloaders import  get_STL10_dataloaders
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_sch

model_checkpoints = ['./SSGANModel/ssgan_20.pt', './SSGANModel/ssgan_40.pt', './SSGANModel/ssgan_60.pt',
                         './SSGANModel/ssgan_80.pt', './SSGANModel/ssgan_100.pt',
                         './SSGANModel/ssgan_120.pt','./SSGANModel/ssgan_140.pt',
                         './SSGANModel/ssgan_150.pt'] 

class linearClassifier(nn.Module):
  def __init__(self, num_classes):
    super(linearClassifier, self).__init__()
    self.discrim = Discriminator(channel = 3, ssup = True, featOnly=True)
    self.fc = nn.Sequential(
                nn.Linear(1024, 1024), 
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(1024, num_classes))
   

  def forward(self, x):
    x = self.discrim(x)
    x = self.fc(x)  
    res = F.log_softmax(x)   
    return res 

betas = (0.0, .9)
lr = 1e-3
num_epochs = 20
_,train_loader, test_loader = get_STL10_dataloaders(batch_size=64)


def train(epoch_p):
    print('Epoch {}/{}'.format(epoch_p, num_epochs - 1))
    print('-' * 10)
    network.train()
    training_loss = 0
    running_corrects = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = network(data)    
        _, preds = torch.max(output, 1)
        loss = F.nll_loss(output, target)
        loss.backward()
        training_loss += loss.item()*data.size(0)
        running_corrects+= torch.sum(preds == target.data)
        optimizer.step()
             
        
    exp_lr_scheduler.step()
    epoch_loss = training_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.item()/len(train_loader.dataset) 

    print('Train Loss: ',epoch_loss,' Acc: ', epoch_acc)

def test():
    network.eval()
  
    correct = 0
    test_loss = 0
    for data, target in test_loader:
        data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda())
        output = network(data)

        
        test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = correct.item()/len(test_loader.dataset) 
    print('\nTest set: Average loss: ',test_loss, 'Accuracy:', test_accuracy)
    
    



for model in model_checkpoints:
    network = linearClassifier(num_classes = 10)
    nn.init.xavier_uniform_(network.fc[0].weight)
    nn.init.xavier_uniform_(network.fc[3].weight)
    network.cuda()
    checkpoint = torch.load(model)
    network.discrim.load_state_dict(checkpoint['dis_state_dict'], strict = False)
    optimizer = optim.Adam(network.parameters(), lr=lr,betas=betas)
    exp_lr_scheduler = lr_sch.StepLR(optimizer, step_size=7, gamma=0.1)
    for param in network.discrim.parameters():
      param.requires_grad = False

    
    for epoch in range(num_epochs):
        train(epoch)

    test()
    #print(test_accuracy)
    #plt.plot(train_loss)
    #plt.savefig('train_loss_plot.png')
