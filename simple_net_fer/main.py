import torch
import torch.nn as nn
from torch.nn import functional as F
import datasets
import argparse
from model import simpleNet
import random

parser = argparse.ArgumentParser("Branch Loss Example")
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr-model', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--use-gpu', type=bool, default=False)
parser.add_argument('--include-neutral', type=bool, default=False)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
args = parser.parse_args()

lr_model = args.lr-model
epochs = args.max-epoch
use_gpu = args.use-gpu
batch_size = args.batch-size
include_neutral = args.include-neutral

num_classes = 8 if include_neutral else 7


def main():
    dataset = datasets.create(
            name='ck+', batch_size=batch_size, use_gpu=use_gpu,
            num_workers=4,
        )
    trainloader, testloader = dataset.trainloader, dataset.testloader

    model = simpleNet(num_classes)
    optimizer_model = torch.optim.SGD(model.parameters(), lr=lr_model, weight_decay=5e-04, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(0, epochs):
        train(model, trainloader, optimizer_model, epoch, criterion=criterion)

        if epoch % 10 == 0:
            test(model, testloader, epoch)

    test(model, testloader, epochs)


def train(model, trainloader, optimizer_model, epoch, criterion=nn.CrossEntropyLoss()):

    model.train()

    for batch_idx, (data, labels) in enumerate(trainloader):
        outputs = model(data)
        loss = criterion(outputs, labels)
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()
    print("current iterating: ", epoch)


def test(model, testloader, after_epoch):

    total, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(testloader):

            outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    acc = correct * 100. / total
    print("accuracy after ", after_epoch, ": ", acc)


if __name__ == '__main__':
    main()
