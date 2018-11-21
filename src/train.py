import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data
from torchvision import datasets, models, transforms

from tensorboardX import SummaryWriter

from data import TransformedDataset, train_test_split_dataset
from logger import write_loss


def train(model, criterion, device, dataloader, optimizer):
    model.train()
    average_loss = 0

    for i, (inputs, labels) in enumerate(dataloader):
        if i % 10 == 0:
            print('Train epoch {} - {}/{}'.format(epoch, i, len(dataloader)))

        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        loss = criterion(outputs, labels.to(device))
        loss.backward()

        average_loss += loss.item() * inputs.size(0)

        optimizer.step()

    return average_loss / len(dataloader.dataset)


def test(model, criterion, device, dataloader):
    model.eval()
    average_loss = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            if i % 10 == 0:
                print('Test epoch {} - {}/{}'.format(epoch, i, len(dataloader)))

            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            average_loss += loss.item() * inputs.size(0)

    return average_loss / len(dataloader.dataset)


if __name__ == '__main__':
    # Training settingmas
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--batch-size', type=int, default=24, metavar='N',
                        help='input batch size for training (default: 24)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    args = parser.parse_args()

    # TODO: Don't hardcode this
    num_classes = 11
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Split the full dataset in a training and validation set
    full_dataset = datasets.ImageFolder('../data/train')
    train_dataset, val_dataset = train_test_split_dataset(full_dataset, test_size=0.2)

    # Create the data loaders for the training and validation set and employ augmentations for the training data
    train_loader = data.DataLoader(TransformedDataset(train_dataset, transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1)),
        # TODO: Add rotations
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])), batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(TransformedDataset(val_dataset, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])), batch_size=args.batch_size, shuffle=False)

    model = models.resnet152(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)



    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9) # TODO: Try using Adam
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    writer = SummaryWriter()
    epoch = 0

    # TODO: Write all params

    while True:
        train_loss = train(model, criterion, device, train_loader, optimizer)
        write_loss(writer, epoch, train_loss, train=True)

        val_loss = test(model, criterion, device, val_loader)
        write_loss(writer, epoch, val_loss, train=False)

        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step()
        epoch += 1

        # TODO: Save best model
