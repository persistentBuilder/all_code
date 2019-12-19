import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
#import datasets
from data_loader import create
import argparse
from model import simpleNet
import random
import time

parser = argparse.ArgumentParser("Branch Loss Example")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--include_neutral', type=bool, default=False)
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
args = parser.parse_args()


def main():
    global args

    args.cuda = torch.cuda.is_available()
    print("using gpu: ", args.cuda)
    batch_size = args.batch_size
    include_neutral = args.include_neutral
    num_classes = 8 if include_neutral else 7
    num_workers = 1 if args.cuda else 0

    dataset = create(name='ck+',batch_size=batch_size, use_gpu=args.cuda, num_workers=num_workers)
    train_loader = dataset.trainloader
    test_loader = dataset.testloader

    model = simpleNet(num_classes)

    if args.cuda:
        model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    for epoch in range(1, args.epochs + 1):
        # train for one epoch
        t = time.time()
        train(train_loader, model, criterion, optimizer, epoch)
        print("time take for epoch ", epoch, ": ", time.time()-t)
        # evaluate on validation set
        acc = test(test_loader, model, criterion, epoch)
        print("-------------------------------------------------------------------")
        print("\n")

        # remember best acc and save checkpoint
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'best_prec1': best_acc,
        # }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):

    correct, total = 0, 0
    loss_triplet = 0.0
    # switch to train mode
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        if args.cuda:
            data = data.cuda()

        data = Variable(data)
        # compute output
        output = model(data)
        loss = criterion(output, label)
        _, predictions = torch.max(output.data, 1)
        loss_triplet += loss.item()
        correct += (predictions == label.data).sum()
        total += data.size()[0]

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("loss  after ", epoch, " epoch: ", loss_triplet)
    acc = correct * 100. / total
    if epoch % 5 == 0:
        print("train accuracy after ", epoch, " epoch: ", acc)


def test(test_loader, model, criterion, epoch):

    correct, total = 0, 0
    loss_triplet = 0
    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            if args.cuda:
                data = data.cuda()

            data = Variable(data)

            # compute output
            output = model(data)

            loss = criterion(output, label)
            loss_triplet += loss.item()
            _, predictions = torch.max(output.data, 1)
            # measure accuracy and record loss
            correct += (predictions == label.data).sum()
            total += data.size()[0]

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


if __name__ == '__main__':
    main()


