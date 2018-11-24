import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data
from torchvision import datasets, models, transforms

import numpy as np

from data import TransformedDataset, train_test_split_dataset, k_fold_split_dataset
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
    # If you're running on Google Colab, uncomment the line below to make argument parsing work.
    # sys.argv = ['foo']

    # Training settingmas
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--batch-size', type=int, default=24, metavar='N',
                        help='input batch size for training (default: 24)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--splits', type=int, default=1,
                        help='number of cross-validation splits (> 2) or 1 for no cross-validation (default: 1)')
    args = parser.parse_args()

    # TODO: Don't hardcode this
    num_classes = 11
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    full_dataset = datasets.ImageFolder('../data/train')

    best_losses = [float('inf')] * args.splits
    best_losses_epoch = [-1] * args.splits

    if args.splits > 2:
        folds = k_fold_split_dataset(full_dataset, n_splits=args.splits)
    else:
        folds = [train_test_split_dataset(full_dataset, test_size=0.2)]

    for fold_idx, (train_dataset, val_dataset) in enumerate(folds):
        print('# STARTING FOLD {}/{}'.format(fold_idx, args.splits))

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
        # TODO: Try using test-time data augmentation

        model = models.resnet152(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)

        # TODO: Try freezing a few of the earlier layers

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9) # TODO: Try using Adam
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # TODO: Write all params

        for epoch in range(args.epochs):
            train_loss = train(model, criterion, device, train_loader, optimizer)
            write_loss(fold_idx, epoch, train_loss, train=True)

            val_loss = test(model, criterion, device, val_loader)
            write_loss(fold_idx, epoch, val_loss, train=False)

            scheduler.step()

            if val_loss < best_losses[fold_idx]:
                best_losses[fold_idx] = val_loss
                best_losses_epoch[fold_idx] = epoch
            # TODO: Save best model
            # torch.save(model.state_dict(), str(epoch) + '.torch')

        print('Completed fold', fold_idx, 'with best loss of', best_losses[fold_idx])

    print('Best loss per fold', best_losses)
    print('Mean:', np.mean(best_losses))
    print('Stddev:', np.std(best_losses))
    print('Epoch for best loss per fold', best_losses_epoch)
