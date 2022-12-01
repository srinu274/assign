import os

import numpy as np
from PIL import Image

import argparse
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
from torch.autograd import grad

from dataset import ColoredMNIST
from model import ConvNet


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('mode', choices=['train-test', 'plotlr'])
  parser.add_argument('--plotpreds', action='store_true')
  parser.add_argument('--resnet18', action='store_true')
  parser.add_argument('--batch_size', default=2000, type=int)
  parser.add_argument('--lr', default=0.001, type=float)
  parser.add_argument('--epochs', default=100, type=int)
  args = parser.parse_args()
  return args


def test_model(model, device, test_loader, set_name="test set"):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device).float()
      output = model(data)
      test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()  # sum up batch loss
      pred = torch.where(torch.gt(output, torch.Tensor([0.0]).to(device)),
                         torch.Tensor([1.0]).to(device),
                         torch.Tensor([0.0]).to(device))  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  print('\nPerformance on {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    set_name, test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

  return 100. * correct / len(test_loader.dataset)


def compute_irm_penalty(losses, dummy):
  g1 = grad(losses[0::2].mean(), dummy, create_graph=True)[0]
  g2 = grad(losses[1::2].mean(), dummy, create_graph=True)[0]
  return (g1 * g2).sum()


def train(model, device, train_loaders, optimizer, epoch):
  model.train()

  train_loaders = [iter(x) for x in train_loaders]

  dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to(device)

  batch_idx = 0
  penalty_multiplier = epoch ** 1.6
  print(f'Using penalty multiplier {penalty_multiplier}')
  while True:
    optimizer.zero_grad()
    error = 0
    penalty = 0
    for loader in train_loaders:
      data, target = next(loader, (None, None))
      if data is None:
        return
      data, target = data.to(device), target.to(device).float()
      output = model(data)
      #print('shape', (output * dummy_w).shape, target.shape)
      loss_erm = F.binary_cross_entropy_with_logits(output * dummy_w, target, reduction='none')
      penalty += compute_irm_penalty(loss_erm, dummy_w)
      error += loss_erm.mean()
    (error + penalty_multiplier * penalty).backward()
    optimizer.step()
    if batch_idx % 2 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tERM loss: {:.6f}\tGrad penalty: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loaders[0]),
               100. * batch_idx / len(train_loaders[0]), error.item(), penalty.item()))
      print('First 20 logits', output.data.cpu().numpy()[:20])

    batch_idx += 1



def train_and_test(epochs=100, batch_size=2000, lr=0.001, plot_pred=False, model_resnet=False):
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  best_train1_acc, best_train2_acc, best_test_acc = 0, 0, 0

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  train1_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='train1',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

  train2_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='train2',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

  test_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='test', transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
    ])),
    batch_size=1000, shuffle=True, **kwargs)

  if model_resnet:
    model = get_resnet18().to(device)
  else:
    model = ConvNet().to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr)

  for epoch in range(1, epochs+1):
    train(model, device, [train1_loader, train2_loader], optimizer, epoch)
    train1_acc = test_model(model, device, train1_loader, set_name='train1 set')
    train2_acc = test_model(model, device, train2_loader, set_name='train2 set')
    test_acc = test_model(model, device, test_loader)
    #if train1_acc > 70 and train2_acc > 70 and test_acc > 60:
    #  print('found acceptable values. stopping training. train1 acc, train2 acc, test_acc: ', train1_acc, train2_acc, test_acc)
    #  if plot_pred:
    #    plot_preds(model, device, test_loader)
    #  return train1_acc, train2_acc, test_acc
    if test_acc > best_test_acc:
      best_test_acc = test_acc
      best_train1_acc = train1_acc
      best_train2_acc = train2_acc

  if plot_pred:
    plot_preds(model, device, test_loader)

  return best_train1_acc, best_train2_acc, best_test_acc


def plot_preds(model, device, test_loader):
  model.eval()
  with torch.no_grad():
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device).float()
    output = model(data)
    pred = torch.where(torch.gt(output, torch.Tensor([0.0]).to(device)),
                        torch.Tensor([1.0]).to(device),
                        torch.Tensor([0.0]).to(device))  # get the index of the max log-probability
  
  fig = plt.figure(figsize=(26, 8))
  columns = 6
  rows = 3
  # ax enables access to manipulate each of subplots
  ax = []

  for i in range(columns * rows):
    img, label = data[i].cpu(), target[i].cpu()
    img = torch.permute(img, (1, 2, 0))
    # create subplot and append to ax
    ax.append(fig.add_subplot(rows, columns, i + 1))
    ax[-1].set_title("Label: " + str(label) + " Pred: "+str(int(pred[i])))  # set title
    plt.imshow(img)

  plt.savefig('preds.png')
  plt.show()  # finally, render the plot


def plot_learning_rate(epochs=10, batch_size=64):
  lrs=[0.0001, 0.001, 0.1, 1]
  train1_accs = []
  train2_accs = []
  test_accs = []
  
  for lr in lrs:
    train1_acc, train2_acc, test_acc = train_and_test(epochs, batch_size, lr)
    train1_accs.append(train1_acc)
    train2_accs.append(train2_acc)
    test_accs.append(test_acc)

  fig, ax = plt.subplots()
  ax.plot(lrs, train1_accs, 'r', label='Train1 acc')
  ax.plot(lrs, train2_accs, 'g', label='Train2 acc')
  ax.plot(lrs, test_accs, 'b', label='Test acc')

  legend = ax.legend(loc='best', shadow=True, fontsize='x-small')

  # Put a nicer background color on the legend.
  legend.get_frame().set_facecolor('C0')


  ax.set_xlabel('Learning rate')
  ax.set_ylabel('Accuracy')

  plt.savefig('lrs.png')
  plt.show()

#train_and_test()

def get_resnet18():
  model_ft = models.resnet18(pretrained=True)
  for name, param in model_ft.named_parameters():
    if 'layer3' in name or 'layer4' in name:
      print('freezing ', name)
      param.requires_grad = False
  
  num_ftrs = model_ft.fc.in_features
  model_ft.fc =  nn.Sequential(
          nn.Linear(num_ftrs, 1),
          nn.Flatten(0, -1)
        )
  print(model_ft)
  return model_ft


if __name__ == "__main__":
  args = parse_args()
  if args.mode == 'train-test':
    train_and_test(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, plot_pred=args.plotpreds, model_resnet=args.resnet18)
  else:
    plot_learning_rate(epochs=args.epochs, batch_size=args.batch_size)