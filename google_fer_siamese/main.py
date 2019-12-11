from losses import TripletLoss
from dataset import SiameseGoogleFer
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import TripletNet, EmbeddingNet


cuda = torch.cuda.is_available()
batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
path = "data/faceexp-comparison-data-train-public.csv"
epochs = 100
lr_model = 0.01

margin = 1.
loss_fn = TripletLoss(margin)
embedding_net = EmbeddingNet()



def main():
    train_dataset = SiameseGoogleFer(path, train_flag=True)
    test_dataset = SiameseGoogleFer(path, train_flag=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    model = TripletNet(embedding_net)
    if cuda:
        model.cuda()
    optimizer_model = torch.optim.SGD(model.parameters(), lr=lr_model, weight_decay=5e-04, momentum=0.9)

    for epoch in range(0, epochs):
        train(model, train_loader, optimizer_model, epoch)

        if epoch % 10 == 0:
            test(model, test_loader, epoch)

    test(model, test_loader, epochs)


def train(model, trainloader, optimizer_model, epoch):

    model.train()

    for batch_idx, (data, labels) in enumerate(trainloader):
        outputs = model(data)
        loss = loss_fn(outputs, labels)
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




