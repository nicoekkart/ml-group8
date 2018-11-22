
# TODO: Create a class Logger that also writes model parameters at the beginning
def write_loss(writer, fold, epoch, loss, train):
    label = 'Train' if train else 'Val'
    #writer.add_scalar(label + '/Loss', loss, epoch)
    print('Fold {} - Epoch {} - {} Loss: {:.4f}'.format(fold, epoch, label, loss))

    # # Create the data loaders for the training and validation set and employ augmentations for the training data
    # train_loader = data.DataLoader(TransformedDataset(train_dataset, transforms.Compose([
    #     transforms.RandomAffine(degrees=30, translate=None, scale=None, shear=10, resample=Image.BILINEAR),
    #     transforms.RandomResizedCrop(224, scale=(0.4, 1)),
    #     # TODO: Add rotations
    #     transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])), batch_size=args.batch_size, shuffle=True)
    # val_loader = data.DataLoader(TransformedDataset(val_dataset, transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])), batch_size=args.batch_size, shuffle=False)
    # # TODO: Try using test-time data augmentation