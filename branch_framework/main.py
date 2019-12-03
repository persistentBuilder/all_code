import torch
import torch.nn as nn
from torch.nn import functional as F
import datasets
import argparse
from model import BranchNet
import random

parser = argparse.ArgumentParser("Branch Loss Example")
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr-model', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=4)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
args = parser.parse_args()

num_classes = 10
branches = int(num_classes/2)
lr_model = args.lr-model
epochs = args.max-epoch
use_gpu = False
random_group = True


def choose_random_label_branch():

    lst = [i for i in range(0, num_classes)]
    random.shuffle(lst)

    labels_to_branch_map = {}
    branch_assigned = 0
    num_of_branch_elements = 2
    curr_branch_elements = 0
    for element in lst:
        labels_to_branch_map[element] = branch_assigned
        curr_branch_elements = curr_branch_elements + 1
        if curr_branch_elements == num_of_branch_elements:
            branch_assigned = branch_assigned + 1
            curr_branch_elements = 0
    return labels_to_branch_map


def main():

    labels_to_branch_map = {6: 0, 9: 0,
                            8: 1, 4: 1,
                            0: 2,
                            2: 3, 7: 3, 1: 3,
                            5: 4, 3: 4}

    if random_group:
        labels_to_branch_map = choose_random_label_branch()
        print("label random, ", labels_to_branch_map)

    dataset = datasets.create(
            name='mnist', batch_size=32, use_gpu=False,
            num_workers=4,
        )
    trainloader, testloader = dataset.trainloader, dataset.testloader

    model = BranchNet(num_classes)
    optimizer_model = torch.optim.SGD(model.parameters(), lr=lr_model, weight_decay=5e-04, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(0, epochs):
        if epoch < epochs/2:
            train(model, trainloader, optimizer_model, epoch, criterion=criterion,
                  labels_to_branch_map=labels_to_branch_map, choose_predefined_branch=True)
        else:
            train(model, trainloader, optimizer_model, epoch, criterion=criterion,
                  labels_to_branch_map=labels_to_branch_map, choose_predefined_branch=False)

        if epoch % 10 == 0:
            test(model, testloader, epoch)

    test(model, testloader, epochs)


def train(model, trainloader, optimizer_model, epoch, criterion=nn.CrossEntropyLoss(), labels_to_branch_map=None,
          choose_predefined_branch=True):

    model.train()

    for batch_idx, (data, labels) in enumerate(trainloader):
        outputs = model(data)
        branch_to_choose_for_label = [labels_to_branch_map[int(x)] for x in labels]

        if not choose_predefined_branch:
            v, w = torch.max(outputs, 2)
            max_values, branch_to_choose_for_label = torch.max(v, 1)

        modified_outputs = torch.stack([outputs[x, branch_to_choose_for_label[x], :] for x in range(0, len(labels))],
                                       dim=0)

        loss = criterion(modified_outputs, labels)
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
            v, w = torch.max(outputs, 2)
            max_values, branch_to_choose_for_label = torch.max(v, 1)
            modified_outputs = torch.stack([outputs[x, branch_to_choose_for_label[x], :] for x in range(0, len(labels))], dim=0)

            predictions = modified_outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    acc = correct * 100. / total
    print("accuracy after ", after_epoch, ": ", acc)


if __name__ == '__main__':
    main()
