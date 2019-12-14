from __future__ import print_function
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from dataset import SiameseGoogleFer
from tripletnet import FECNet, EmbeddNet
from losses import TripletLoss
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch siamese triplet loss for google fer')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='TripletNet', type=str,
                    help='name of experiment')

best_acc = 0


def main():
    global args, best_acc
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    batch_size = 32
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
    train_path = "data/faceexp-comparison-data-train-public.csv"
    test_path = "data/faceexp-comparison-data-test-public.csv"
    train_dataset = SiameseGoogleFer(train_path, train_flag=True, transform=transform)
    test_dataset = SiameseGoogleFer(test_path, train_flag=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    model = EmbeddNet()
    tnet = FECNet(model)
    tnet.embeddingnet.resnet.requires_grad = False

    if args.cuda:
        tnet.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            tnet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    criterion = TripletLoss(margin=args.margin)
    #optimizer = optim.SGD(tnet.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = torch.optim.Adam(tnet.parameters(), lr=args.lr)

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    for epoch in range(1, args.epochs + 1):
        # train for one epoch
        train(train_loader, tnet, criterion, optimizer, epoch)
        # evaluate on validation set
        acc = test(test_loader, tnet, criterion, epoch)

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': tnet.state_dict(),
        #     'best_prec1': best_acc,
        # }, is_best)


def train(train_loader, tnet, criterion, optimizer, epoch):

    correct, total = 0, 0
    loss_triplet = 0.0
    # switch to train mode
    tnet.train()
    for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()

        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        dista, distb, distc, embedded_x, embedded_y, embedded_z = tnet(data1, data2, data3)
        # 1 means, dista should be larger than distb

        loss = criterion(embedded_x, embedded_y, embedded_z, size_average=True) + \
                criterion(embedded_y, embedded_x, embedded_z, size_average=True)
        loss_triplet += loss.item()

        correct += triplet_correct(dista.detach().numpy(), distb.detach().numpy(), distc.detach().numpy())
        total += data1.size()[0]

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("loss  after ", epoch, " epoch: ", loss_triplet)
    acc = correct * 100. / total
    if epoch % 5 == 0:
        print("train accuracy after ", epoch, " epoch: ", acc)


def test(test_loader, tnet, criterion, epoch):

    correct, total = 0, 0
    # switch to evaluation mode
    tnet.eval()
    with torch.no_grad():
        for batch_idx, (data1, data2, data3) in enumerate(test_loader):
            if args.cuda:
                data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
            data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

            # compute output
            dista, distb, distc, embedded_x, embedded_y, embedded_z = tnet(data1, data2, data3)

            loss_triplet = criterion(embedded_x, embedded_y, embedded_z, size_average=False) + \
                           criterion(embedded_y, embedded_x, embedded_z, size_average=False)

            # measure accuracy and record loss
            correct += triplet_correct(dista.detach().numpy(), distb.detach().numpy(), distc.detach().numpy())
            total += data1.size()[0]

    acc = correct * 100. / total
    print("test accuracy after ", epoch, " epoch: ", acc)
    return acc


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')


def triplet_correct(dista, distb, distc):
    return np.logical_and(dista < distb, dista < distc).sum()


if __name__ == '__main__':
    main()    
